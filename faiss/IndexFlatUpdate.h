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
#include <unordered_map>
#include <cstring>
#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>

namespace faiss {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatUpdateCodes : Index {
    size_t code_size;
    idx_t nremove = 0; // 被标记删除的点数量
    idx_t labelcount = 0; // 记录已添加点的最大label
    std::unordered_map<int, int> label_lookup_; // label->id
    std::vector<bool> is_deleted;
    std::vector<idx_t> label; // id->label
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

    void find_vector(const float* x){
        for (int i = 0; i < ntotal;i++){
            float *base = new float[d];
            memcpy(base, &codes.data()[i * d * sizeof(float)], sizeof(float) * d);
            for (int j = 0; j < d;j++){
                if(x[j]!=base[j])
                    break;
                if(j==d-1&&x[j]==base[j]){
                    printf("find vector, id:%d, label:%ld\n", i, label[i]);
                    return;
                }
            }
            delete[] base;
        }
        printf("vector not found\n");
        return;
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
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
