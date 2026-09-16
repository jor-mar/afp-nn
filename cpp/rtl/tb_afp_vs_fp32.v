// tb_afp_vs_fp32.v
//
// Loads AFP-encoded and raw-FP32 test vectors (produced by
// gen_testvectors.cpp) for the same underlying data, runs both
// afp_dot_product_top and fp32_mac_behavioral, and reports:
//   - both computed dot-product values (and their agreement with the
//     expected value written by the generator)
//   - both pipelines' clock-cycle counts, and the resulting speedup
//
// Run with (after `pip install`-free `apt-get install iverilog` or
// building Icarus Verilog / Verilator locally -- see README.md, this
// could not be executed in the sandbox that produced this repository):
//
//   iverilog -g2012 -o sim afp_element_decode.v afp_block_dot.v \
//       afp_dot_product_top.v fp32_mac_behavioral.v tb_afp_vs_fp32.v
//   vvp sim

`timescale 1ns/1ps

module tb_afp_vs_fp32;
    localparam NUM_BLOCKS = 8;
    localparam NUM_ELEMENTS = NUM_BLOCKS * 16;

    reg clk = 0;
    reg rst_n = 0;
    reg start_afp = 0, start_fp32 = 0;
    wire done_afp, done_fp32;
    wire signed [63:0] afp_mantissa;
    wire signed [15:0] afp_exp;
    wire [31:0] afp_cycles, fp32_cycles;
    wire real fp32_result;

    always #5 clk = ~clk; // 100 MHz

    afp_dot_product_top #(.NUM_BLOCKS(NUM_BLOCKS)) u_afp (
        .clk(clk), .rst_n(rst_n), .start(start_afp), .done(done_afp),
        .result_mantissa_o(afp_mantissa), .result_exp_o(afp_exp),
        .cycle_count_o(afp_cycles)
    );

    fp32_mac_behavioral #(.NUM_ELEMENTS(NUM_ELEMENTS)) u_fp32 (
        .clk(clk), .rst_n(rst_n), .start(start_fp32), .done(done_fp32),
        .result_o(fp32_result), .cycle_count_o(fp32_cycles)
    );

    real afp_result_real;
    real expected_result;
    integer expected_file, r;

    initial begin
        // Load AFP block test vectors (hierarchical load into the DUT's
        // internal "memory" arrays -- standard Icarus/Verilator testbench
        // practice for simple synchronous-ROM-style memories).
        $readmemh("testvectors/afp_a_fields.hex", u_afp.afp_a_fields);
        $readmemh("testvectors/afp_b_fields.hex", u_afp.afp_b_fields);
        $readmemh("testvectors/afp_a_exp.hex",    u_afp.afp_a_exp);
        $readmemh("testvectors/afp_b_exp.hex",    u_afp.afp_b_exp);

        // Load the same underlying values' raw FP32 bit patterns.
        $readmemh("testvectors/fp32_a.hex", u_fp32.a_mem);
        $readmemh("testvectors/fp32_b.hex", u_fp32.b_mem);

        expected_file = $fopen("testvectors/expected_dot.txt", "r");
        if (expected_file == 0) begin
            $display("ERROR: could not open testvectors/expected_dot.txt -- run gen_testvectors first.");
            $finish;
        end
        r = $fscanf(expected_file, "%f", expected_result);
        $fclose(expected_file);

        rst_n = 0;
        #20;
        rst_n = 1;
        #10;

        // ---- Run AFP pipeline -------------------------------------------
        @(posedge clk);
        start_afp = 1;
        @(posedge clk);
        start_afp = 0;
        wait (done_afp);
        afp_result_real = afp_mantissa * (2.0 ** afp_exp);

        // ---- Run behavioral FP32 pipeline --------------------------------
        @(posedge clk);
        start_fp32 = 1;
        @(posedge clk);
        start_fp32 = 0;
        wait (done_fp32);

        // ---- Report ----------------------------------------------------
        $display("========================================================");
        $display("AFP  vs FP32 dot-product RTL comparison (%0d elements)", NUM_ELEMENTS);
        $display("========================================================");
        $display("Expected (software, float64) : %.6f", expected_result);
        $display("AFP  RTL result               : %.6f  (mantissa=%0d exp=%0d)",
                  afp_result_real, afp_mantissa, afp_exp);
        $display("FP32 RTL result (behavioral)  : %.6f", fp32_result);
        $display("--------------------------------------------------------");
        $display("AFP  cycles : %0d  (%0d elements/cycle throughput)",
                  afp_cycles, NUM_ELEMENTS / afp_cycles);
        $display("FP32 cycles : %0d  (%0d elements/cycle throughput)",
                  fp32_cycles, NUM_ELEMENTS / fp32_cycles);
        $display("Speedup (FP32 cycles / AFP cycles): %.2fx", fp32_cycles * 1.0 / afp_cycles);
        $display("--------------------------------------------------------");
        if ((afp_result_real - expected_result) < 0.05 * (expected_result < 0 ? -expected_result : expected_result) + 0.01 &&
            (expected_result - afp_result_real) < 0.05 * (expected_result < 0 ? -expected_result : expected_result) + 0.01)
            $display("PASS: AFP RTL result matches expected value within tolerance.");
        else
            $display("FAIL: AFP RTL result does not match expected value.");
        $finish;
    end

    initial begin
        $dumpfile("tb_afp_vs_fp32.vcd");
        $dumpvars(0, tb_afp_vs_fp32);
    end
endmodule
