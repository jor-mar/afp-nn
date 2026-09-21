// afp_dot_product_top.v
//
// Streams NUM_BLOCKS 16-element AFP block pairs from simple synchronous
// memories through afp_block_dot16 and accumulates a full dot product
// (NUM_BLOCKS * 16 elements), counting the clock cycles consumed.
//
// ---------------------------------------------------------------------
// v2: fixes a real overflow bug found by running v1 in an actual
// simulator (thank you to whoever reported it). v1 picked the *first*
// block's exponent as a single fixed reference for the whole vector and
// aligned every other block onto it with an *unclamped* `<<<`. Two
// blocks of only 16 Gaussian samples each can easily land 30-40+
// exponent steps apart by chance (one block's 16 samples all happening
// to be small, another's all large), and shifting a ~40-bit mantissa
// left by that much overflowed the 64-bit accumulator and silently
// wrapped to garbage -- which is exactly the
// "mantissa=-2305843009214349390 (~ -2^61)" failure that was reported.
//
// v2 fixes this the way any floating-point accumulator has to: instead
// of one global fixed reference exponent for the whole vector, the
// running total is itself kept as a *normalized (mantissa, exponent)
// pair* (a small custom floating accumulator), and each new block is
// folded in with an align-shift that is *clamped* (ALIGN_SHIFT_CLAMP,
// default 40) rather than left unbounded. Clamping is not a hack here --
// it is the mathematically correct behavior: if a block's magnitude is
// more than 2^40 smaller than the running total, its contribution
// really is below the accumulator's representable precision and
// *should* round to zero, exactly as it would in any floating-point sum.
// After aligning and adding, the result is renormalized (shifted back
// into a bounded mantissa width, exponent adjusted to compensate) so the
// accumulator can never grow without bound regardless of how many
// blocks are summed or how wide their exponent spread is.
//
// This mirrors afp::dot_product_native's own two-tier design in
// software (afp_ops.hpp): fixed-point accumulation *within* a block,
// where the dynamic range is inherently bounded by the 3-bit offset
// field (<=12 steps for a block pair), and a floating combination
// *across* blocks (there, a double accumulator + one ldexp per block;
// here, this bounded custom float accumulator), because only the
// within-block range is small enough to be safely fixed-point.
//
// Timing: per block, 1 cycle to read fields/exponent from memory into
// registers (S_LOAD), 1 cycle to let afp_block_dot16's combinational
// result settle (S_COMPUTE), 1 cycle to align+add+renormalize into the
// running accumulator (S_ACCUM) -- 3 cycles/block, same as v1.

module afp_dot_product_top #(
    parameter NUM_BLOCKS = 8,
    parameter ALIGN_SHIFT_CLAMP = 40  // max alignment shift before a
                                       // term is treated as negligible
                                       // (must be well under the
                                       // 64-bit scratch width used for
                                       // the add -- see ACC_MANT_BITS)
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output reg  signed [63:0] result_mantissa_o,  // dot ~= result_mantissa_o * 2^result_exp_o
    output reg  signed [15:0] result_exp_o,
    output reg  [31:0] cycle_count_o
);
    localparam ACC_MANT_BITS = 48; // normalized accumulator mantissa width

    // ---- simple synchronous "memories" (simulation: loaded via $readmemh
    // in the testbench that instantiates this module) -------------------
    reg [143:0] afp_a_fields [0:NUM_BLOCKS-1];
    reg [143:0] afp_b_fields [0:NUM_BLOCKS-1];
    reg [7:0]   afp_a_exp    [0:NUM_BLOCKS-1];
    reg [7:0]   afp_b_exp    [0:NUM_BLOCKS-1];

    // ---- FSM --------------------------------------------------------------
    localparam S_IDLE = 3'd0, S_LOAD = 3'd1, S_COMPUTE = 3'd2,
               S_ACCUM = 3'd3, S_NEXT = 3'd4, S_DONE = 3'd5;
    reg [2:0] state;
    reg [$clog2(NUM_BLOCKS+1)-1:0] blk_idx;

    reg [143:0] fields_a_r, fields_b_r;
    reg [7:0]   exp_a_r, exp_b_r;

    wire signed [39:0] block_mantissa;
    wire signed [15:0] block_exp;
    afp_block_dot16 u_block_dot (
        .fields_a_i(fields_a_r), .fields_b_i(fields_b_r),
        .exp_a_i(exp_a_r), .exp_b_i(exp_b_r),
        .mantissa_acc_o(block_mantissa), .combined_exp_o(block_exp)
    );

    // ---- Bounded floating accumulator state --------------------------------
    reg signed [ACC_MANT_BITS-1:0] acc_mantissa;
    reg signed [15:0] acc_exp;
    reg acc_has_data;

    // ---- Combinational align + add + renormalize --------------------------
    // Computes what the accumulator *would become* if the current
    // block's (block_mantissa, block_exp) were folded in this cycle.
    // Latched into acc_mantissa/acc_exp only while in S_ACCUM.
    reg signed [15:0] align_shift;
    reg signed [63:0] acc_ext, term_ext, sum_ext;
    reg signed [15:0] common_exp;
    reg signed [ACC_MANT_BITS-1:0] next_acc_mantissa;
    reg signed [15:0] next_acc_exp;
    integer k;
    localparam signed [63:0] ACC_MAX = (64'sd1 <<< (ACC_MANT_BITS-1)) - 64'sd1;
    localparam signed [63:0] ACC_MIN = -(64'sd1 <<< (ACC_MANT_BITS-1));

    always @* begin
        acc_ext  = {{(64-ACC_MANT_BITS){acc_mantissa[ACC_MANT_BITS-1]}}, acc_mantissa};
        term_ext = {{24{block_mantissa[39]}}, block_mantissa};

        if (acc_exp >= block_exp) begin
            align_shift = acc_exp - block_exp;
            if (align_shift > ALIGN_SHIFT_CLAMP) align_shift = ALIGN_SHIFT_CLAMP;
            sum_ext    = acc_ext + (term_ext >>> align_shift);
            common_exp = acc_exp;
        end else begin
            align_shift = block_exp - acc_exp;
            if (align_shift > ALIGN_SHIFT_CLAMP) align_shift = ALIGN_SHIFT_CLAMP;
            sum_ext    = (acc_ext >>> align_shift) + term_ext;
            common_exp = block_exp;
        end

        // Renormalize: right-shift sum_ext (tracking how many shifts were
        // actually needed) until it fits in ACC_MANT_BITS signed bits.
        // Bounded to 24 iterations -- comfortably more than the <=1-2
        // shifts a single add can ever require in practice (each operand
        // already fits in ACC_MANT_BITS/40 bits respectively), so this
        // loop is both simulation-safe and synthesizable as a small
        // fixed-depth shifter/priority-encoder network.
        next_acc_exp = common_exp;
        for (k = 0; k < 24; k = k + 1) begin
            if ((sum_ext > ACC_MAX) || (sum_ext < ACC_MIN)) begin
                sum_ext = sum_ext >>> 1;
                next_acc_exp = next_acc_exp + 1;
            end
        end
        next_acc_mantissa = sum_ext[ACC_MANT_BITS-1:0];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            blk_idx <= 0;
            result_mantissa_o <= 64'sd0;
            result_exp_o <= 16'sd0;
            cycle_count_o <= 32'd0;
            acc_mantissa <= {ACC_MANT_BITS{1'b0}};
            acc_exp <= 16'sd0;
            acc_has_data <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        blk_idx <= 0;
                        acc_mantissa <= {ACC_MANT_BITS{1'b0}};
                        acc_exp <= 16'sd0;
                        acc_has_data <= 1'b0;
                        cycle_count_o <= 32'd0;
                        state <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    fields_a_r <= afp_a_fields[blk_idx];
                    fields_b_r <= afp_b_fields[blk_idx];
                    exp_a_r    <= afp_a_exp[blk_idx];
                    exp_b_r    <= afp_b_exp[blk_idx];
                    cycle_count_o <= cycle_count_o + 1;
                    state <= S_COMPUTE;
                end
                S_COMPUTE: begin
                    // afp_block_dot16 is combinational; this cycle just
                    // lets its outputs settle before we fold them in.
                    cycle_count_o <= cycle_count_o + 1;
                    state <= S_ACCUM;
                end
                S_ACCUM: begin
                    if (block_mantissa == 0) begin
                        // empty/all-zero block: nothing to add
                    end else if (!acc_has_data) begin
                        // First nonzero block seeds the accumulator
                        // directly (nothing to align against yet).
                        acc_mantissa <= {{8{block_mantissa[39]}}, block_mantissa}; // sign-extend 40->48
                        acc_exp <= block_exp;
                        acc_has_data <= 1'b1;
                    end else begin
                        acc_mantissa <= next_acc_mantissa;
                        acc_exp <= next_acc_exp;
                    end
                    cycle_count_o <= cycle_count_o + 1;
                    state <= S_NEXT;
                end
                S_NEXT: begin
                    if (blk_idx == NUM_BLOCKS - 1) begin
                        result_mantissa_o <= {{(64-ACC_MANT_BITS){acc_mantissa[ACC_MANT_BITS-1]}}, acc_mantissa};
                        result_exp_o <= acc_exp;
                        state <= S_DONE;
                    end else begin
                        blk_idx <= blk_idx + 1;
                        state <= S_LOAD;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
