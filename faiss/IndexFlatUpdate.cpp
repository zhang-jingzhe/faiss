/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexFlatUpdate.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/sorting.h>
#include <faiss/utils/utils.h>
#include <cstring>

#include <faiss/impl/CodePacker.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

IndexFlatUpdate::IndexFlatUpdate(idx_t d, MetricType metric)
        : IndexFlatUpdateCodes(sizeof(float) * d, d, metric) {}

void IndexFlatUpdate::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT(k > 0);

    // we see the distances and labels as heaps
    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res, sel);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);
    } else if (is_similarity_metric(metric_type)) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, get_xb(), d, n, ntotal, metric_type, metric_arg, &res);
    } else {
        FAISS_THROW_IF_NOT(!sel);
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, get_xb(), d, n, ntotal, metric_type, metric_arg, &res);
    }
}

void IndexFlatUpdate::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    IDSelector* sel = params ? params->sel : nullptr;

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result, sel);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlatUpdate::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

namespace {

struct FlatL2Dis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final {
        ndis++;
        return fvec_L2sqr(q, (float*)code, d);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return fvec_L2sqr(b + j * d, b + i * d, d);
    }

    explicit FlatL2Dis(const IndexFlatUpdate& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_L2sqr_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }
};

struct FlatIPDis : FlatCodesDistanceComputer {
    size_t d;
    idx_t nb;
    const float* q;
    const float* b;
    size_t ndis;

    float symmetric_dis(idx_t i, idx_t j) final override {
        return fvec_inner_product(b + j * d, b + i * d, d);
    }

    float distance_to_code(const uint8_t* code) final override {
        ndis++;
        return fvec_inner_product(q, (const float*)code, d);
    }

    explicit FlatIPDis(const IndexFlatUpdate& storage, const float* q = nullptr)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              d(storage.d),
              nb(storage.ntotal),
              q(q),
              b(storage.get_xb()),
              ndis(0) {}

    void set_query(const float* x) override {
        q = x;
    }

    // compute four distances
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) final override {
        ndis += 4;

        // compute first, assign next
        const float* __restrict y0 =
                reinterpret_cast<const float*>(codes + idx0 * code_size);
        const float* __restrict y1 =
                reinterpret_cast<const float*>(codes + idx1 * code_size);
        const float* __restrict y2 =
                reinterpret_cast<const float*>(codes + idx2 * code_size);
        const float* __restrict y3 =
                reinterpret_cast<const float*>(codes + idx3 * code_size);

        float dp0 = 0;
        float dp1 = 0;
        float dp2 = 0;
        float dp3 = 0;
        fvec_inner_product_batch_4(q, y0, y1, y2, y3, d, dp0, dp1, dp2, dp3);
        dis0 = dp0;
        dis1 = dp1;
        dis2 = dp2;
        dis3 = dp3;
    }
};

} // namespace

FlatCodesDistanceComputer* IndexFlatUpdate::get_FlatCodesDistanceComputer() const {
    if (metric_type == METRIC_L2) {
        return new FlatL2Dis(*this);
    } else if (metric_type == METRIC_INNER_PRODUCT) {
        return new FlatIPDis(*this);
    } else {
        return get_extra_distance_computer(
                d, metric_type, metric_arg, ntotal, get_xb());
    }
}

void IndexFlatUpdate::reconstruct(idx_t key, float* recons) const {
    memcpy(recons, &(codes[key * code_size]), code_size);
}

void IndexFlatUpdate::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, x, sizeof(float) * d * n);
    }
}

void IndexFlatUpdate::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
        memcpy(x, bytes, sizeof(float) * d * n);
    }
}

IndexFlatUpdateCodes::IndexFlatUpdateCodes(size_t code_size, idx_t d, MetricType metric)
        : Index(d, metric), code_size(code_size) {}

IndexFlatUpdateCodes::IndexFlatUpdateCodes() : code_size(0) {}

void IndexFlatUpdateCodes::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size);
    is_deleted.resize(ntotal + n, false);
    idx_t pos = 0;
    if (nremove > 0) {
        while (!deleted_elements.empty()){
            idx_t id_replaced = *deleted_elements.begin();
            deleted_elements.erase(id_replaced);
            sa_encode(1, &x[pos], codes.data() + (id_replaced * code_size));
            nremove--;
            pos++;
            is_deleted[id_replaced] = false;
            if(pos == n || nremove == 0){
                break;
            }
        }
        if(nremove==0){
            sa_encode(n - pos, &x[pos], codes.data() + (ntotal * code_size));
            ntotal += (n - pos);
        }
    }
    else{
        sa_encode(n, x, codes.data() + (ntotal * code_size));
        ntotal += n;
    }
}

void IndexFlatUpdateCodes::reset() {
    codes.clear();
    is_deleted.clear();
    ntotal = 0;
    nremove = 0;
}

size_t IndexFlatUpdateCodes::sa_code_size() const {
    return code_size;
}

size_t IndexFlatUpdateCodes::mark_deleted(const IDSelectorArray& sel) {
    nremove += sel.n;
    for (int i = 0; i < sel.n;i++){
        FAISS_THROW_IF_NOT(is_deleted[sel.ids[i]] == false);
        is_deleted[sel.ids[i]] = true;
        deleted_elements.insert(sel.ids[i]);
    }
    return nremove;
}

void IndexFlatUpdateCodes::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    sa_decode(ni, codes.data() + i0 * code_size, recons);
}

void IndexFlatUpdateCodes::reconstruct(idx_t key, float* recons) const {
    reconstruct_n(key, 1, recons);
}

FlatCodesDistanceComputer* IndexFlatUpdateCodes::get_FlatCodesDistanceComputer()
        const {
    FAISS_THROW_MSG("not implemented");
}

CodePacker* IndexFlatUpdateCodes::get_CodePacker() const {
    return new CodePackerFlat(code_size);
}

} // namespace faiss
