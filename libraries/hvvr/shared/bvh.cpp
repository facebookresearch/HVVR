/**
 * Copyright (c) 2017-present, Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bvh.h"
#include "constants_math.h"
#include "debug.h"
#include "graphics_types.h"
#include "model.h"
#include "vector_math.h"

#include <algorithm>
#include <vector>

namespace hvvr {

//==============================================================================
// Bounding box data structures.
//==============================================================================

struct Box {
    vector3 lower;
    vector3 upper;
    uint32_t offset;
    uint32_t size;

    Box() {}
    Box(const vector3& lower, const vector3& upper) : lower(lower), upper(upper), offset(0), size(0) {}
    Box(const vector3& lower, const vector3& upper, uint32_t offset, uint32_t size)
        : lower(lower), upper(upper), offset(offset), size(size) {}

    Box operator|(const Box& a) const {
        return Box(min(lower, a.lower), max(upper, a.upper),
                   // not sure if any code in here is reliant on the below behavior, but I'm preserving it from the SSE
                   // implementation, just in case
                   min(offset, a.offset), max(size, a.size));
    }
    Box& operator|=(const Box& a) {
        return *this = *this | a;
    }
    static Box empty() {
        return Box(vector3(Infinity), vector3(-Infinity));
    }
    float surfaceArea() const {
        vector3 extents = upper - lower;
        return extents.x * (extents.y + extents.z) + extents.y * extents.z;
    }
};

struct Box2 {
    Box boxes[2];

    uint32_t& offset(size_t i) {
        return boxes[i].offset;
    }
    const uint32_t& offset(size_t i) const {
        return boxes[i].offset;
    }
    uint32_t& size(size_t i) {
        return boxes[i].size;
    }
    const uint32_t& size(size_t i) const {
        return boxes[i].size;
    }
    Box bound() const {
        return boxes[0] | boxes[1];
    }
};

struct BoxPlus {
    Box box;
    uint32_t firstTriangle;
    uint32_t boundTriangle;
    uint32_t offset;
    uint32_t depth;
};

struct Box4 {
    BoxPlus boxes[4];
};

//==============================================================================
// Bounding volume hierarchy evalution/debugging.
//==============================================================================

struct BoxEvaluate {
    void operator()(const Box2* node) {
        totalDepth = maxDepth = numNode2s = numTriangles = 0;
        traverse(node, 0);
    }
    void traverse(const Box2* node, uint32_t depth) {
        totalDepth += depth;
        if (maxDepth < depth)
            maxDepth = depth;
        numNode2s++;
        if (node->offset(0))
            traverse(node + node->offset(0), depth + 1);
        else
            numTriangles += node->size(0);
        if (node->offset(1))
            traverse(node + node->offset(1), depth + 1);
        else
            numTriangles += node->size(1);
    }
    uint32_t totalDepth;
    uint32_t maxDepth;
    uint32_t numNode2s;
    uint32_t numTriangles;
};

struct EvaluateBVH4 {
    void operator()(const Box4* node) {
        totalDepth = maxDepth = numNode4s = numTriangles = 0;
        traverse(node, 0);
    }
    void traverse(const Box4* node, uint32_t depth) {
        totalDepth += depth;
        if (maxDepth < depth)
            maxDepth = depth;
        numNode4s++;
        for (size_t k = 0; k < 4; k++) {
            if (node->boxes[k].offset)
                traverse(node + node->boxes[k].offset, depth + 1);
            else
                numTriangles += node->boxes[k].boundTriangle - node->boxes[k].firstTriangle;
        }
    }
    uint32_t totalDepth;
    uint32_t maxDepth;
    uint32_t numNode4s;
    uint32_t numTriangles;
};

struct EvaluateTopology {
    void operator()(const TopologyNode* node) {
        totalDepth = maxDepth = numNode4s = numTriangles = 0;
        traverse(node, 0);
    }
    void traverse(const TopologyNode* node, uint32_t depth) {
        totalDepth += depth;
        if (maxDepth < depth)
            maxDepth = depth;
        numNode4s++;
        for (size_t k = 0; k < 4; k++) {
            if (node->isLeaf(k))
                numTriangles += node->getBoundTriangleIndex(k) - node->getFirstTriangleIndex(k);
            else
                traverse(node + node->getChildOffset(k), depth + 1);
        }
    }
    uint32_t totalDepth;
    uint32_t maxDepth;
    uint32_t numNode4s;
    uint32_t numTriangles;
};

//==============================================================================
// Bounding volume hierarchy construction routines.
// Traditional top-down SAH optimal divisive build algorithm.
//==============================================================================

// Scans along one of 3 axis, looking for a minmum cost split.
bool scanAxis(size_t& bestIndex, float& bestCost, float* temp, const Box** boxes, size_t size, int axis) {
    // Sort the leaf boxes on the axis.
    std::sort(boxes, boxes + size, [axis](const Box*& a, const Box*& b) {
        return a->lower[axis] + a->upper[axis] <
               b->lower[axis] + b->upper[axis];
    });

    // Scan in reverse.
    Box box = Box::empty();
    for (size_t i = 1; i < size; i++) {
        box |= *boxes[size - i];
        temp[size - i] = box.surfaceArea() * i;
    }

    // Scan forward and evaluate SAH cost.
    bool newAxis = false;
    box = Box::empty();
    for (size_t i = 1; i < size; i++) {
        box |= *boxes[i - 1];
        float cost = temp[i] + box.surfaceArea() * i;
        if (bestCost > cost) {
            bestCost = cost;
            bestIndex = i;
            newAxis = true;
        }
    }
    return newAxis;
}

size_t helpBuildBVH2(Box2* node, float* temp, const Box** boxes, size_t size) {
    if (size < 2)
        return 0;

    // Look along each axis to determine which contains the lowest cost split.
    int axis = 0;
    size_t index = 0;
    float cost = Infinity;
    scanAxis(index, cost, temp, boxes, size, 0);
    if (scanAxis(index, cost, temp, boxes, size, 1))
        axis = 1;
    if (scanAxis(index, cost, temp, boxes, size, 2))
        axis = 2;

    // Sort the boxes along the axis with the best split.
    std::sort(boxes, boxes + size, [axis](const Box*& a, const Box*& b) {
        return a->lower[axis] + a->upper[axis] <
               b->lower[axis] + b->upper[axis];
    });

    // Recurse on the lower half of the split range.
    size_t offset = 1;
    if (size_t count = helpBuildBVH2(node + offset, temp, boxes, index)) {
        node->boxes[0] = node[offset].bound();
        node->offset(0) = (uint32_t)offset;
        offset += count;
    } else {
        node->boxes[0] = *boxes[0];
        for (size_t i = 1; i < index; i++)
            node->boxes[0] |= *boxes[i];
        node->offset(0) = 0;
    }
    node->size(0) = (uint32_t)index;

    // Recurse on the upper half of the split range.
    if (size_t count = helpBuildBVH2(node + offset, temp + index, boxes + index, size - index)) {
        node->boxes[1] = node[offset].bound();
        node->offset(1) = (uint32_t)offset;
        offset += count;
    } else {
        node->boxes[1] = *boxes[index];
        for (size_t i = index + 1; i < size; i++)
            node->boxes[1] |= *boxes[i];
        node->offset(1) = 0;
    }
    node->size(1) = (uint32_t)(size - index);

    // Should we subdivide further? This controls the balance of tree depth vs leaf size.
    // Check to see if this split costs more than just intersecting the triangles directly.
    // Left side is the cost to intersect the triangles, right side is the cost to intersect the children.
    const float nodeIntersectMultiplier = 2.0f; // intersecting a node costs the same as 2 triangles
                                                // size < cost / node->bound().surfaceArea() + nodeIntersectMultiplier
    return node->bound().surfaceArea() * (float(size) - nodeIntersectMultiplier) < cost ? 0 : offset;
}

std::vector<Box2> buildBVH2(MeshData& mesh) {
    size_t numTriangles = mesh.triShade.size();

    // Initialize the leaf boxes and pointers.
    std::vector<const Box*> leafBoxPointers(numTriangles);
    std::vector<Box> leafBoxes(numTriangles);
    for (size_t i = 0; i < numTriangles; i++) {
        const PrecomputedTriangleShade& triShade = mesh.triShade[i];
        vector3 p0 = mesh.verts[triShade.indices[0]].pos;
        vector3 p1 = mesh.verts[triShade.indices[1]].pos;
        vector3 p2 = mesh.verts[triShade.indices[2]].pos;
        leafBoxes[i].lower = min(p0, p1, p2);
        leafBoxes[i].upper = max(p0, p1, p2);
        leafBoxPointers[i] = &leafBoxes[i];
    }

    // Recursively build the BVH.
    std::vector<float> scratchpad(numTriangles);
    std::vector<Box2> node2s(numTriangles - 1);
    size_t count = helpBuildBVH2(node2s.data(), scratchpad.data(), leafBoxPointers.data(), numTriangles);
    if (!count) {
        Box box = leafBoxes[0];
        for (size_t i = 1; i < numTriangles; i++)
            box |= leafBoxes[i];
        node2s[0].boxes[0] = Box::empty();
        node2s[0].offset(0) = 0;
        node2s[0].size(0) = 0;
        node2s[0].boxes[1] = box;
        node2s[0].offset(1) = 0;
        node2s[0].size(1) = uint32_t(numTriangles);
        count = 1;
    }
    node2s.resize(count);

    // Sort the triangles to reflect the newly sorted order of the leaf boxes.
    std::vector<PrecomputedTriangleShade> sortedTriangles(numTriangles);
    for (size_t i = 0; i < numTriangles; i++) {
        ptrdiff_t index = leafBoxPointers[i] - leafBoxes.data();
        sortedTriangles[i] = mesh.triShade[index];
    }
    mesh.triShade = std::move(sortedTriangles);

    return node2s;
}

//==============================================================================
// Normalization routine for BVH2s: repetetively eliminates nodes until there are either <= 4
// leaves or the number of leaves is a number that will divide evenly into a 4-wide bvh.
// Note: we're not presently using this path, but rather allowing odd-sized trees.
//==============================================================================

struct NormalizeBVH2ForConversion {
    size_t operator()(Box2* nodes, size_t numNodes) {
        if (numNodes < 3)
            fail("Can't normalize BVHs with fewer than 3 nodes.");
        if (numNodes % 3 == 0)
            return numNodes;
        // Traverse the hierarchy to find the smallest cost node to eliminate.
        // Kill a child pointer (the node will automatically adopt the leaves of that child).
        minCost = Infinity;
        minCostOffset = nullptr;
        traverse(nodes);
        *minCostOffset = 0;
        return (*this)(nodes, numNodes - 1);
    }

    bool traverse(Box2* node) {
        // Post-order traversal: visit the children first.
        for (uint32_t i = 0; i < 2; i++)
            if (node->offset(i))
                if (traverse(node + node->offset(i)))
                    minCostOffset = &node->offset(i);
        // Check to see if this node contains two sets of triangles.
        if (node->offset(0) || node->offset(1))
            return false;
        // Calculate the cost of replacing this node with a list of triangles.
        float cost = node->bound().surfaceArea() * (node->size(0) + node->size(1));
        // If this isn't the cheapest cost return false.
        if (minCost <= cost)
            return false;
        // This is the cheapest leaf, update our cost and return true.
        minCost = cost;
        return true;
    }
    float minCost;
    uint32_t* minCostOffset;
};

//==============================================================================
// 2-BVH -> 4-BVH conversion
//==============================================================================

uint32_t helpBuildBVH4(Box4*& out, const Box2* node, BoxPlus* parentCache, uint32_t first) {
    // Post-order traversal: visit the children first.
    BoxPlus cache[6];
    uint32_t cacheSize = 0;
    if (node->offset(0))
        cacheSize += helpBuildBVH4(out, node + node->offset(0), cache + cacheSize, first);
    else
        cache[cacheSize++] = BoxPlus{node->boxes[0], first, first + node->size(0), 0, 0};
    if (node->offset(1))
        cacheSize += helpBuildBVH4(out, node + node->offset(1), cache + cacheSize, first + node->size(0));
    else
        cache[cacheSize++] =
            BoxPlus{node->boxes[1], first + node->size(0), first + node->size(0) + node->size(1), 0, 0};

    // If we get fewer than 4 nodes from our children, pass them to our parent.
    if (cacheSize < 4) {
        for (uint32_t i = 0; i < cacheSize; i++)
            parentCache[i] = cache[i];
        return cacheSize;
    }

    // We got 4 or more candidates from
    uint32_t j0Best = 4;
    uint32_t j1Best = 5;
    if (cacheSize > 4) {
        // Cache-size more than 4, find the set of 4 that produce the smallest box and pass the rest.
        // Try to minimize the total depth by favoring passing nodes with higher depth.
        std::sort(cache, cache + cacheSize, [](const BoxPlus& a, const BoxPlus& b) { return a.depth > b.depth; });
        float minCost = Infinity;
        if (cacheSize == 5) {
            for (uint32_t j0 = 0; j0 < 5; j0++) {
                Box box = Box::empty();
                for (uint32_t i = 0; i < 5; i++)
                    if (i != j0)
                        box |= cache[i].box;
                if (minCost > box.surfaceArea()) {
                    minCost = box.surfaceArea();
                    j0Best = j0;
                }
            }
        } else {
            for (uint32_t j1 = 1; j1 < 6; j1++)
                for (uint32_t j0 = 0; j0 < j1; j0++) {
                    Box box = Box::empty();
                    for (uint32_t i = 0; i < 6; i++)
                        if (i != j0 && i != j1)
                            box |= cache[i].box;
                    if (minCost > box.surfaceArea()) {
                        minCost = box.surfaceArea();
                        j0Best = j0;
                        j1Best = j1;
                    }
                }
        }
    }

    // Allocate a new box from the output list.
    BoxPlus* outBoxes = (--out)->boxes;
    // Add the box we just created to our parent's cache.
    BoxPlus* parent = parentCache++;
    *parent = BoxPlus{Box::empty(), 0, 0, (uint32_t)(out - (Box4*)nullptr), 0};
    for (uint32_t i = 0; i < cacheSize; i++) {
        if (i == j0Best || i == j1Best) {
            // Push the evected boxes into the parent's cache.
            *parentCache++ = cache[i];
            continue;
        }
        // Add these children to the allocated box and update the bounding box.
        if (cache[i].offset)
            cache[i].offset -= (uint32_t)(out - (Box4*)nullptr);
        if (cache[i].depth + 1 > parent->depth)
            parent->depth = cache[i].depth + 1;
        parent->box |= cache[i].box;
        *outBoxes++ = cache[i];
    }
    // Sort each box such that the leaves are first and nodes of the same type are sorted by surface area.
    std::sort(out->boxes, out->boxes + 4, [](const BoxPlus& a, const BoxPlus& b) {
        return (a.offset == 0 && b.offset != 0) ||
               ((a.offset == 0 || b.offset != 0) && a.box.surfaceArea() > b.box.surfaceArea());
    });
    // Return the total number of boxes we added to our parent's cache (our cache - 4 + 1 (the new node)).
    return cacheSize - 3;
}

std::vector<Box4> buildBVH4(std::vector<Box2>& node2s) {
    std::vector<Box4> nodePlus4s((node2s.size() + 2) / 3);

    // Collapse the 2-BVH to a 4-BVH.  Note: out is updated by the helper function.
    Box4* out = nodePlus4s.data() + nodePlus4s.size();
    BoxPlus cache[3];
    uint32_t cacheSize = helpBuildBVH4(out, node2s.data(), cache, 0);

    // Check to see if we need to process the root separately.
    if (cacheSize == 1) {
        if (nodePlus4s.data() != out)
            fail("Internal consistency LogFatalure with respect to BVH4 normalization.");
        return nodePlus4s;
    }

    // Allocate a new box from the output list.
    BoxPlus* outBoxes = (--out)->boxes;
    if (nodePlus4s.data() != out)
        fail("Internal consistency LogFatalure with respect to BVH4 normalization.");

    // Add some empty boxes to fill up the output;
    BoxPlus empty = {};
    for (uint32_t extra = 4 - cacheSize; extra; --extra)
        *outBoxes++ = empty;

    // Add these boxes to the root.
    for (uint32_t i = 0; i < cacheSize; i++) {
        // Update the cache offsets.
        if (cache[i].offset)
            cache[i].offset -= (uint32_t)(out - (Box4*)nullptr);
        *outBoxes++ = cache[i];
    }

    // Sort each box such that the leaves are first and nodes of the same type are sorted by surface area.
    std::sort(out->boxes, out->boxes + 4, [](const BoxPlus& a, const BoxPlus& b) {
        return (a.offset == 0 && b.offset != 0) ||
               ((a.offset == 0 || b.offset != 0) && a.box.surfaceArea() > b.box.surfaceArea());
    });

    return nodePlus4s;
}

//==============================================================================
// Topology Extraction
//==============================================================================

// Takes the nodes in the order that the BVH4 builder constructed them and produces them in depth first search order.
struct TopologyBuilder {
    void build(MeshData& mesh, const std::vector<Box4>& in) {
        triangles.reserve(mesh.triShade.size());
        topologyNodes.reserve(in.size());
        srcTriangles = mesh.triShade.data();

        // Generate both the topology and the new indices.
        traverse(in.data());
        if (triangles.size() != mesh.triShade.size())
            fail("Didn't copy all triangles.");
        if (topologyNodes.size() != in.size())
            fail("Didn't copy all vertices.");
        mesh.nodes = std::move(topologyNodes);

        // Sort the vertices so they're in the order that BVH update will read them.
        uint32_t vertexCount = uint32_t(mesh.verts.size());
        std::vector<ShadingVertex> vertices(vertexCount);
        uint32_t invalidIndex = uint32_t(vertexCount);
        uint32_t newIndex = invalidIndex;
        std::vector<uint32_t> vertexOffsets(vertexCount, invalidIndex);
        for (ptrdiff_t t = triangles.size() - 1; t >= 0; t--) {
            for (int v = 2; v >= 0; v--) {
                uint32_t index = triangles[t].indices[v];

                if (vertexOffsets[index] == invalidIndex) {
                    vertexOffsets[index] = --newIndex;
                    vertices[newIndex] = mesh.verts[index];
                }

                triangles[t].indices[v] = vertexOffsets[index];
            }
        }
        if (newIndex)
            fail("Didn't copy all vertices.");

        mesh.triShade = std::move(triangles);
        mesh.verts = std::move(vertices);
    }
    size_t traverse(const Box4* in) {
        // Generate a new node.
        size_t index = topologyNodes.size();
        topologyNodes.emplace_back();
        TopologyNode& out = topologyNodes.back();
        out.leafMask = 0;
        out.setFirstTriangleIndex(0, uint32_t(triangles.size()));

        // Iterate over the boxes and initialize out new node.
        for (size_t k = 0; k < 4; k++) {
            const BoxPlus& box = in->boxes[k];
            if (box.offset) {
                // This box bounds a sub-tree.
                out.setChildOffset(k, (uint32_t)(traverse(in + box.offset) - index));
                continue;
            }

            // This box bounds a collection of triangles.
            out.leafMask |= 1u << k;
            for (size_t i = box.firstTriangle, e = box.boundTriangle; i != e; i++)
                triangles.emplace_back(srcTriangles[i]);
            out.setBoundTriangleIndex(k, uint32_t(triangles.size()));
        }
        return index;
    }
    std::vector<PrecomputedTriangleShade> triangles;
    const PrecomputedTriangleShade* srcTriangles;
    std::vector<TopologyNode> topologyNodes;
};

bool GenerateTopology(MeshData& mesh) {
    // Build the 2-wide BVH.
    std::vector<Box2> node2s = buildBVH2(mesh);
    BoxEvaluate()(node2s.data());

    // Build the 4-wide bvh.
    std::vector<Box4> node4s = buildBVH4(node2s);
    EvaluateBVH4()(node4s.data());

    // Build the topology.
    TopologyBuilder().build(mesh, node4s);
    EvaluateTopology()(mesh.nodes.data());

    return true;
}

} // namespace hvvr
