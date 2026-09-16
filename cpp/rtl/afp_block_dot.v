// afp_block_dot.v
//
// Computes the dot product of one 16-element AFP block pair (A . B),
// implementing the paper's Section 3 formula
//
//   a.b = 2^(ea*+eb*) * sum_i (-1)^(sa_i xor sb_i) * 2^-(ta_i+tb_i) * ma_i * mb_i
//
// as a fixed-point integer pipeline: per-lane 6x6-bit signed significand
// multiply, alignment via left-shift-to-the-block's-max-shift (so every
// lane is a safe left shift, never a lossy right shift -- the same trick
// afp::dot_product_native uses in software, see afp_ops.hpp), a 16-way
// adder tree, and a single combined-exponent output. This mirrors
// Figure 10 of "Be Like Water": "decode -> shift by offset sum -> integer
// multiply-accumulate -> scale by shared exponent."
//
// Result convention: dot_product = mantissa_acc_o * 2^combined_exp_o
// (a software/testbench-side ldexp(mantissa_acc_o, combined_exp_o)
// reconstructs the float value; afp_dot_product_top.v does this across
// multiple blocks to accumulate a full dot product).
//
// Purely combinational; synthesizable (a single pipeline register stage
// can be inserted at the module boundary by the instantiating design if a
// higher clock frequency is required -- afp_dot_product_top.v registers
// its output already).

module afp_block_dot16 (
    input  wire [143:0] fields_a_i,   // 16 x 9-bit private fields, element 0 in [8:0]
    input  wire [143:0] fields_b_i,
    input  wire [7:0]   exp_a_i,      // shared exponent byte (biased, +127), block A
    input  wire [7:0]   exp_b_i,      // shared exponent byte (biased, +127), block B
    output reg  signed [39:0] mantissa_acc_o,
    output reg  signed [15:0] combined_exp_o
);
    integer i;

    // ---- per-lane decode -----------------------------------------------
    wire        sign_a   [0:15];
    wire [5:0]  sig_a    [0:15];
    wire [3:0]  shift_a  [0:15];
    wire        zero_a   [0:15];
    wire        sign_b   [0:15];
    wire [5:0]  sig_b    [0:15];
    wire [3:0]  shift_b  [0:15];
    wire        zero_b   [0:15];

    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin : DECODE
            afp_element_decode dec_a (
                .field_i(fields_a_i[g*9 +: 9]),
                .sign_o(sign_a[g]), .significand_o(sig_a[g]),
                .shift_o(shift_a[g]), .is_zero_o(zero_a[g])
            );
            afp_element_decode dec_b (
                .field_i(fields_b_i[g*9 +: 9]),
                .sign_o(sign_b[g]), .significand_o(sig_b[g]),
                .shift_o(shift_b[g]), .is_zero_o(zero_b[g])
            );
        end
    endgenerate

    // ---- per-lane shift sum + block-wide max shift ----------------------
    reg [4:0] shift_sum [0:15];  // 0..12, 5 bits for margin
    reg [4:0] max_shift;
    always @* begin
        max_shift = 5'd0;
        for (i = 0; i < 16; i = i + 1) begin
            shift_sum[i] = shift_a[i] + shift_b[i];
            if (!zero_a[i] && !zero_b[i] && shift_sum[i] > max_shift)
                max_shift = shift_sum[i];
        end
    end

    // ---- per-lane signed product, aligned by left-shift ------------------
    reg signed [31:0] lane_term [0:15];
    reg signed [12:0] raw_prod;  // unsigned 6x6 -> 12 bits, +1 guard
    always @* begin
        for (i = 0; i < 16; i = i + 1) begin
            raw_prod = $signed({1'b0, sig_a[i]}) * $signed({1'b0, sig_b[i]});
            if (zero_a[i] || zero_b[i]) begin
                lane_term[i] = 32'sd0;
            end else if (sign_a[i] ^ sign_b[i]) begin
                lane_term[i] = -({{19{1'b0}}, raw_prod} <<< (max_shift - shift_sum[i]));
            end else begin
                lane_term[i] =  ({{19{1'b0}}, raw_prod} <<< (max_shift - shift_sum[i]));
            end
        end
    end

    // ---- 16-way adder tree (behavioral sum; synthesizes to a tree) -------
    always @* begin
        mantissa_acc_o = 40'sd0;
        for (i = 0; i < 16; i = i + 1)
            mantissa_acc_o = mantissa_acc_o + lane_term[i];
        // dot = mantissa_acc_o * 2^(ea+eb - 254 - 10 - max_shift)
        //   -254 removes both bytes' IEEE-754 bias (127 each);
        //   -10   removes the Q5.5 (implicit-one + 5-bit mantissa) scale
        //         each of the two 6-bit significands contributes;
        //   -max_shift undoes the common left-shift alignment applied above.
        combined_exp_o = $signed({8'b0, exp_a_i}) + $signed({8'b0, exp_b_i})
                        - 16'sd254 - 16'sd10 - {11'b0, max_shift};
    end

endmodule
