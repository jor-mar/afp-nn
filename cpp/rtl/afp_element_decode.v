// afp_element_decode.v
//
// Decodes one AFP element's 9-bit private field (1 sign + 3 offset +
// 5 mantissa -- the baseline "AFP8" layout from afp_core.hpp, without the
// positive-field/zero-field bonus-bit optimizations; see cpp/rtl/README.md
// for why this RTL scopes to the baseline layout) into the three pieces
// the MAC datapath needs:
//
//   sign_o        : 0 = positive, 1 = negative
//   significand_o : 6-bit unsigned, implicit leading one folded in for
//                   normalized values ({1'b1, mantissa}); for denormal
//                   values (offset == 3'b111) the leading bit is 0
//                   ({1'b0, mantissa}), matching afp_scalar_to_float's
//                   handling in afp_codec.hpp.
//   shift_o       : effective right-shift (0..6) this element's
//                   contribution needs relative to the block's shared
//                   exponent. For normalized values this is simply the
//                   3-bit offset field (0..6). For denormal values
//                   (offset == 7) this RTL uses a fixed shift of 6 --
//                   the same "denormals share the smallest *normalized*
//                   bucket's reference exponent (e*-6)" convention
//                   documented in afp_codec.hpp's encode_block() comment
//                   for the software codec, kept consistent here so a
//                   block encoded with pos/zero-field disabled
//                   (afp::EncodeOptions with both flags false) decodes
//                   identically in hardware and software.
//   is_zero_o     : 1 if this element is exact zero (offset==7, mantissa==0)
//
// Purely combinational; synthesizable.

module afp_element_decode (
    input  wire [8:0] field_i,      // {sign, offset[2:0], mantissa[4:0]}
    output wire        sign_o,
    output wire [5:0]  significand_o,
    output wire [3:0]  shift_o,
    output wire        is_zero_o
);
    wire        sign_w    = field_i[8];
    wire [2:0]  offset_w  = field_i[7:5];
    wire [4:0]  mantissa_w = field_i[4:0];
    wire        denormal_w = (offset_w == 3'b111);

    assign sign_o        = sign_w;
    assign significand_o = denormal_w ? {1'b0, mantissa_w} : {1'b1, mantissa_w};
    assign shift_o        = denormal_w ? 4'd6 : {1'b0, offset_w};
    assign is_zero_o      = denormal_w & (mantissa_w == 5'b00000);

endmodule
