// fp32_mac_behavioral.v
//
// A *behavioral, simulation-only* FP32 scalar multiply-accumulate
// reference, used purely to produce an apples-to-apples clock-cycle count
// against afp_dot_product_top.v. This module is NOT synthesizable (it
// uses Verilog `real` arithmetic) -- a real IEEE-754 FPU is a substantial
// design in its own right and is out of scope here; see cpp/rtl/README.md
// for why a behavioral reference is the honest, standard way to produce a
// cycle-count comparison without building a full synthesizable FPU, and
// for the same caveat already given in the C++ software benchmark's
// output about what this comparison does and doesn't demonstrate.
//
// Model: one scalar multiply-accumulate per clock cycle (a conservative,
// favorable-to-FP32 assumption -- many real FP32 multipliers need several
// pipeline stages per operation; we still charge FP32 only 1 cycle/element
// here). AFP's block datapath instead processes all 16 elements of a
// block through a parallel integer-multiplier array combinationally, at a
// fixed ~2-cycles-per-block overhead (see afp_dot_product_top.v) -- the
// resulting cycle-count gap is the RTL's demonstration of AFP's higher
// compute density per Section 3.8 of the paper, at the same clock.

module fp32_mac_behavioral #(
    parameter NUM_ELEMENTS = 128
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,
    output real result_o,
    output reg  [31:0] cycle_count_o
);
    reg [31:0] a_mem [0:NUM_ELEMENTS-1];
    reg [31:0] b_mem [0:NUM_ELEMENTS-1];

    function real bits_to_real;
        input [31:0] bits;
        reg          sign;
        reg [7:0]    exp;
        reg [22:0]   frac;
        real         val;
        begin
            sign = bits[31];
            exp  = bits[30:23];
            frac = bits[22:0];
            if (exp == 8'd0) begin
                val = (frac / 8388608.0) * (2.0 ** -126);
            end else begin
                val = (1.0 + frac / 8388608.0) * (2.0 ** (real'(exp) - 127.0));
            end
            bits_to_real = sign ? -val : val;
        end
    endfunction

    localparam S_IDLE = 2'd0, S_RUN = 2'd1, S_DONE = 2'd2;
    reg [1:0] state;
    reg [$clog2(NUM_ELEMENTS+1)-1:0] idx;
    real acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            idx <= 0;
            acc <= 0.0;
            cycle_count_o <= 32'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        idx <= 0;
                        acc <= 0.0;
                        cycle_count_o <= 32'd0;
                        state <= S_RUN;
                    end
                end
                S_RUN: begin
                    // One scalar multiply-accumulate per cycle.
                    acc <= acc + bits_to_real(a_mem[idx]) * bits_to_real(b_mem[idx]);
                    cycle_count_o <= cycle_count_o + 1;
                    if (idx == NUM_ELEMENTS - 1) begin
                        state <= S_DONE;
                    end else begin
                        idx <= idx + 1;
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

    assign result_o = acc;

endmodule
