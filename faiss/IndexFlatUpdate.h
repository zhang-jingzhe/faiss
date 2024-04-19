/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef INDEX_FLAT_H
#define INDEX_FLAT_H

#pragma once

#include <vector>
#include <set>
#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>

namespace faiss {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatUpdateCodes : Index {
    size_t code_size;
    idx_t nremove;  // 被标记删除的点数量
    std::vector<bool> is_deleted;
    std::set<idx_t> deleted_elements;
    /// encoded dataset, size ntotal * code_size
    std::vector<uint8_t> codes;

    IndexFlatUpdateCodes();

    IndexFlatUpdateCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);

    /// default add uses sa_encode
    void add(idx_t n, const float* x) override;

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    size_t sa_code_size() const override;

    size_t mark_deleted(const IDSelectorArray& sel);

    /** a FlatCodesDistanceComputer offers a distance_to_code method */
    virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }

    // returns a new instance of a CodePacker
    CodePacker* get_CodePacker() const;
};

/** Index that stores the full vectors and performs exhaustive search */
struct IndexFlatUpdate : IndexFlatUpdateCodes {
    explicit IndexFlatUpdate(
            idx_t d, ///< dimensionality of the input vectors
            MetricType metric = METRIC_L2);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            const idx_t* labels) const;

    // get pointer to the floating point data
    float* get_xb() {
        return (float*)codes.data();
    }
    const float* get_xb() const {
        return (const float*)codes.data();
    }

    IndexFlatUpdate() {}

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* The stanadlone codec interface (just memcopies in this case) */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss

#endif
