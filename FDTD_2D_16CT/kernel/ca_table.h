#ifndef __CA_TABLE_HPP__
#define __CA_TABLE_HPP__


constexpr int num_entries_ca_table = 4;
alignas(aie::vector_decl_align) const float ca_table_ab[] = {
    0.4823, 0.4823, 0.0525, 0.0525,
    0.4823, 0.4823, 0.0525, 0.0525
};
alignas(aie::vector_decl_align) const float ca_table_cd[] = {
    0.4823, 0.4823, 0.0525, 0.0525,
    0.4823, 0.4823, 0.0525, 0.0525
};
#endif
