// afp_dot_product_top.v
//
// Streams NUM_BLOCKS 16-element AFP block pairs from simple synchronous
// memories through afp_block_dot16 and accumulates a full dot product
// (NUM_BLOCKS * 16 elements), counting the clock cycles consumed.
//
// Per block: 1 cycle to read the block's fields/exponent from memory into
// registers, 1 cycle for afp_block_dot16's combinational result to be
// latched (afp_block_dot16 itself is pure combinational logic, so in a
// real pipelined design this could be fused into the same cycle as the
// read -- we keep them separate stages here for a conservative, easy-to-
// read two-stage pipeline: LOAD -> COMPUTE+ACCUMULATE), so DONE fires
// after 2*NUM_BLOCKS + 2 cycles. That is the throughput number this RTL's
// testbench compares against the scalar FP32 reference's 1-cycle-per-
// element loop -- see cpp/rtl/README.md for the full cycle-count writeup.
//
// Cross-block accumulation: each block produces (mantissa_acc, exponent)
// with dot_block = mantissa_acc * 2^exponent. The very first nonzero
// block's exponent is latched as `ref_exp`; every block's mantissa is
// then re-aligned onto that fixed reference via a signed arithmetic shift
// (matching software's ldexp-based combination in afp::dot_product_native,
// just fixed at hardware synthesis time to a single global shift register
// instead of a floating re-normalization) before being summed into a wide
// (64-bit) fixed-point accumulator.

module afp_dot_product_top #(
    parameter NUM_BLOCKS = 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output reg  signed [63:0] result_mantissa_o,  // dot = result_mantissa_o * 2^result_exp_o
    output reg  signed [15:0] result_exp_o,
    output reg  [31:0] cycle_count_o
);
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

    reg signed [15:0] ref_exp;
    reg ref_exp_set;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            blk_idx <= 0;
            result_mantissa_o <= 64'sd0;
            result_exp_o <= 16'sd0;
            cycle_count_o <= 32'd0;
            ref_exp_set <= 1'b0;
            ref_exp <= 16'sd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        blk_idx <= 0;
                        result_mantissa_o <= 64'sd0;
                        ref_exp_set <= 1'b0;
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
                    // lets its outputs settle before we latch/accumulate.
                    cycle_count_o <= cycle_count_o + 1;
                    state <= S_ACCUM;
                end
                S_ACCUM: begin
                    if (!ref_exp_set && block_mantissa != 0) begin
                        ref_exp <= block_exp;
                        ref_exp_set <= 1'b1;
                        result_mantissa_o <= result_mantissa_o + {{24{block_mantissa[39]}}, block_mantissa};
                    end else if (block_mantissa == 0) begin
                        // empty/all-zero block: nothing to add
                    end else begin
                        // Align this block's mantissa onto ref_exp via a
                        // signed shift (left if this block's own exponent
                        // is larger than the reference, right otherwise).
                        if (block_exp >= ref_exp)
                            result_mantissa_o <= result_mantissa_o +
                                ({{24{block_mantissa[39]}}, block_mantissa} <<< (block_exp - ref_exp));
                        else
                            result_mantissa_o <= result_mantissa_o +
                                ({{24{block_mantissa[39]}}, block_mantissa} >>> (ref_exp - block_exp));
                    end
                    cycle_count_o <= cycle_count_o + 1;
                    state <= S_NEXT;
                end
                S_NEXT: begin
                    if (blk_idx == NUM_BLOCKS - 1) begin
                        result_exp_o <= ref_exp;
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
