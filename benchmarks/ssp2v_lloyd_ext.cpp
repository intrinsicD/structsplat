#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kSuccess = 0;
constexpr int kInvalidArgument = 1;
constexpr int kArithmeticError = 2;
constexpr int kBufferTooSmall = 3;
constexpr int kInternalError = 4;
constexpr std::uint32_t kMaximumSweeps = 100;

thread_local std::string g_last_error;

class ArithmeticError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class BufferTooSmall final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::uint32_t RotateRight(const std::uint32_t value, const unsigned shift) {
  return (value >> shift) | (value << (32 - shift));
}

std::array<std::uint8_t, 32> Sha256(const std::vector<std::uint8_t>& input) {
  constexpr std::array<std::uint32_t, 64> kRound = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
  };
  std::array<std::uint32_t, 8> state = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  if (input.size() > std::numeric_limits<std::uint64_t>::max() / 8) {
    throw ArithmeticError("SHA-256 input length overflows uint64 bits");
  }
  std::vector<std::uint8_t> message = input;
  const auto bit_length = static_cast<std::uint64_t>(input.size()) * 8;
  message.push_back(0x80u);
  while (message.size() % 64 != 56) {
    message.push_back(0u);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(static_cast<std::uint8_t>(bit_length >> shift));
  }

  for (std::size_t offset = 0; offset < message.size(); offset += 64) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const auto base = offset + 4 * index;
      words[index] = (static_cast<std::uint32_t>(message[base]) << 24) |
                     (static_cast<std::uint32_t>(message[base + 1]) << 16) |
                     (static_cast<std::uint32_t>(message[base + 2]) << 8) |
                     static_cast<std::uint32_t>(message[base + 3]);
    }
    for (std::size_t index = 16; index < 64; ++index) {
      const auto s0 = RotateRight(words[index - 15], 7) ^
                      RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3);
      const auto s1 = RotateRight(words[index - 2], 17) ^
                      RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    auto a = state[0];
    auto b = state[1];
    auto c = state[2];
    auto d = state[3];
    auto e = state[4];
    auto f = state[5];
    auto g = state[6];
    auto h = state[7];
    for (std::size_t index = 0; index < 64; ++index) {
      const auto sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const auto choose = (e & f) ^ ((~e) & g);
      const auto temporary1 = h + sum1 + choose + kRound[index] + words[index];
      const auto sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  std::array<std::uint8_t, 32> digest{};
  for (std::size_t index = 0; index < state.size(); ++index) {
    digest[4 * index] = static_cast<std::uint8_t>(state[index] >> 24);
    digest[4 * index + 1] = static_cast<std::uint8_t>(state[index] >> 16);
    digest[4 * index + 2] = static_cast<std::uint8_t>(state[index] >> 8);
    digest[4 * index + 3] = static_cast<std::uint8_t>(state[index]);
  }
  return digest;
}

void AppendU16Le(std::vector<std::uint8_t>& output, const std::uint16_t value) {
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8));
}

void AppendU32Le(std::vector<std::uint8_t>& output, const std::uint32_t value) {
  output.push_back(static_cast<std::uint8_t>(value));
  output.push_back(static_cast<std::uint8_t>(value >> 8));
  output.push_back(static_cast<std::uint8_t>(value >> 16));
  output.push_back(static_cast<std::uint8_t>(value >> 24));
}

struct Vec {
  std::array<std::int32_t, 3> value{};

  bool operator==(const Vec& other) const { return value == other.value; }
  bool operator<(const Vec& other) const { return value < other.value; }
};

struct WeightedVec {
  Vec vector;
  std::uint64_t count = 0;
};

struct Spec {
  std::uint8_t family = 0;
  std::uint8_t matched_stages = 0;
  std::uint16_t base_k = 0;
  std::uint8_t stage_index = 0;
  std::size_t codebook_size = 0;

  bool StoresUnsignedBytes() const { return stage_index == 0; }
};

void ValidateSpec(const Spec& spec, const std::uint8_t start_id) {
  if (spec.family > 1) {
    throw std::invalid_argument("family_id must be zero or one");
  }
  if (spec.matched_stages != 2 && spec.matched_stages != 3) {
    throw std::invalid_argument("matched stage count must be two or three");
  }
  if (spec.base_k == 0 || spec.codebook_size == 0) {
    throw std::invalid_argument("base_k and codebook size must be positive");
  }
  if (start_id > 3) {
    throw std::invalid_argument("start_id must lie in [0,3]");
  }
  if (spec.family == 0) {
    const auto expected = static_cast<std::size_t>(2 * spec.matched_stages - 1) * spec.base_k;
    if (spec.stage_index != 0 || spec.codebook_size != expected) {
      throw std::invalid_argument("flat stage metadata or codebook size is inconsistent");
    }
  } else if (spec.stage_index >= spec.matched_stages ||
             spec.codebook_size != spec.base_k) {
    throw std::invalid_argument("RVQ stage metadata or codebook size is inconsistent");
  }
}

std::uint64_t SquaredDistance(const Vec& left, const Vec& right) {
  std::uint64_t total = 0;
  for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
    const auto difference = static_cast<std::int64_t>(left.value[coordinate]) -
                            static_cast<std::int64_t>(right.value[coordinate]);
    const auto magnitude = difference < 0
                               ? static_cast<std::uint64_t>(-(difference + 1)) + 1
                               : static_cast<std::uint64_t>(difference);
    const auto square = magnitude * magnitude;  // an int32 difference square fits uint64
    if (square > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - total) {
      throw ArithmeticError("an exact squared distance exceeds signed int64");
    }
    total += square;
  }
  return total;
}

std::vector<WeightedVec> Collapse(const std::int32_t* rows, const std::size_t row_count,
                                  const Spec& spec) {
  if (rows == nullptr || row_count == 0) {
    throw std::invalid_argument("input vectors must be nonempty");
  }
  std::vector<Vec> sorted(row_count);
  std::array<std::int32_t, 3> minima = {
      std::numeric_limits<std::int32_t>::max(),
      std::numeric_limits<std::int32_t>::max(),
      std::numeric_limits<std::int32_t>::max(),
  };
  std::array<std::int32_t, 3> maxima = {
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::min(),
  };
  for (std::size_t index = 0; index < row_count; ++index) {
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      sorted[index].value[coordinate] = rows[3 * index + coordinate];
      minima[coordinate] = std::min(minima[coordinate], sorted[index].value[coordinate]);
      maxima[coordinate] = std::max(maxima[coordinate], sorted[index].value[coordinate]);
    }
    if (spec.StoresUnsignedBytes() &&
        (sorted[index].value[0] < 0 || sorted[index].value[0] > 255 ||
         sorted[index].value[1] < 0 || sorted[index].value[1] > 255 ||
         sorted[index].value[2] < 0 || sorted[index].value[2] > 255)) {
      throw std::invalid_argument("flat/RVQ-stage-0 vectors must lie in [0,255]");
    }
  }
  std::sort(sorted.begin(), sorted.end());
  std::vector<WeightedVec> weighted;
  weighted.reserve(sorted.size());
  for (const auto& vector : sorted) {
    if (weighted.empty() || !(weighted.back().vector == vector)) {
      weighted.push_back({vector, 1});
    } else {
      if (weighted.back().count == std::numeric_limits<std::uint64_t>::max()) {
        throw ArithmeticError("unique-vector count overflows uint64");
      }
      ++weighted.back().count;
    }
  }
  Vec minimum;
  Vec maximum;
  minimum.value = minima;
  maximum.value = maxima;
  // The coordinate-wise bounding-box diagonal bounds every possible distance.
  SquaredDistance(minimum, maximum);
  return weighted;
}

std::array<std::uint8_t, 32> StartKey(const Vec& vector, const Spec& spec,
                                      const std::uint8_t start_id) {
  static constexpr char kDomain[] = "structsplat-comp011-vq-start-v1";
  std::vector<std::uint8_t> bytes;
  bytes.reserve(sizeof(kDomain) + 18);
  bytes.insert(bytes.end(), kDomain, kDomain + sizeof(kDomain));  // includes one NUL
  bytes.push_back(spec.family);
  bytes.push_back(spec.matched_stages);
  AppendU16Le(bytes, spec.base_k);
  bytes.push_back(spec.stage_index);
  bytes.push_back(start_id);
  for (const auto component : vector.value) {
    AppendU32Le(bytes, static_cast<std::uint32_t>(component));
  }
  return Sha256(bytes);
}

bool KeyVectorLess(const std::size_t left, const std::size_t right,
                   const std::vector<std::array<std::uint8_t, 32>>& keys,
                   const std::vector<WeightedVec>& vectors) {
  if (keys[left] != keys[right]) {
    return keys[left] < keys[right];
  }
  return vectors[left].vector < vectors[right].vector;
}

void ValidateCentroids(const std::vector<Vec>& codebook, const Spec& spec) {
  for (const auto& centroid : codebook) {
    for (const auto component : centroid.value) {
      if (spec.StoresUnsignedBytes()) {
        if (component < 0 || component > 255) {
          throw ArithmeticError("stage-zero centroid is not uint8-representable");
        }
      } else if (component < std::numeric_limits<std::int16_t>::min() ||
                 component > std::numeric_limits<std::int16_t>::max()) {
        throw ArithmeticError("later RVQ centroid is not int16-representable");
      }
    }
  }
}

std::vector<Vec> Initialize(const std::vector<WeightedVec>& vectors, const Spec& spec,
                            const std::uint8_t start_id) {
  std::vector<std::array<std::uint8_t, 32>> keys;
  keys.reserve(vectors.size());
  for (const auto& vector : vectors) {
    keys.push_back(StartKey(vector.vector, spec, start_id));
  }
  std::size_t first = 0;
  for (std::size_t index = 1; index < vectors.size(); ++index) {
    if (KeyVectorLess(index, first, keys, vectors)) {
      first = index;
    }
  }
  std::vector<bool> selected(vectors.size(), false);
  std::vector<std::uint64_t> nearest(vectors.size());
  std::vector<std::size_t> order = {first};
  selected[first] = true;
  for (std::size_t index = 0; index < vectors.size(); ++index) {
    nearest[index] = SquaredDistance(vectors[index].vector, vectors[first].vector);
  }
  const auto unique_target = std::min(vectors.size(), spec.codebook_size);
  while (order.size() < unique_target) {
    bool found = false;
    std::size_t chosen = 0;
    for (std::size_t index = 0; index < vectors.size(); ++index) {
      if (selected[index]) {
        continue;
      }
      if (!found || nearest[index] > nearest[chosen] ||
          (nearest[index] == nearest[chosen] && KeyVectorLess(index, chosen, keys, vectors))) {
        chosen = index;
        found = true;
      }
    }
    if (!found) {
      throw std::logic_error("farthest-first failed to select a remaining vector");
    }
    order.push_back(chosen);
    selected[chosen] = true;
    for (std::size_t index = 0; index < vectors.size(); ++index) {
      nearest[index] = std::min(
          nearest[index], SquaredDistance(vectors[index].vector, vectors[chosen].vector));
    }
  }

  std::vector<Vec> codebook;
  codebook.reserve(spec.codebook_size);
  for (const auto index : order) {
    codebook.push_back(vectors[index].vector);
  }
  if (vectors.size() < spec.codebook_size) {
    std::vector<std::size_t> cyclic(vectors.size());
    for (std::size_t index = 0; index < cyclic.size(); ++index) {
      cyclic[index] = index;
    }
    std::sort(cyclic.begin(), cyclic.end(), [&](const auto left, const auto right) {
      return KeyVectorLess(left, right, keys, vectors);
    });
    std::size_t cursor = 0;
    while (codebook.size() < spec.codebook_size) {
      codebook.push_back(vectors[cyclic[cursor % cyclic.size()]].vector);
      ++cursor;
    }
  }
  std::sort(codebook.begin(), codebook.end());
  ValidateCentroids(codebook, spec);
  return codebook;
}

struct Assignment {
  std::vector<std::size_t> indices;
  std::uint64_t objective = 0;
};

std::size_t Nearest(const Vec& vector, const std::vector<Vec>& codebook,
                    std::uint64_t* distance_output = nullptr) {
  std::size_t best = 0;
  auto best_distance = SquaredDistance(vector, codebook[0]);
  for (std::size_t index = 1; index < codebook.size(); ++index) {
    const auto distance = SquaredDistance(vector, codebook[index]);
    if (distance < best_distance) {
      best = index;
      best_distance = distance;
    }
  }
  if (distance_output != nullptr) {
    *distance_output = best_distance;
  }
  return best;
}

Assignment Assign(const std::vector<WeightedVec>& weighted,
                  const std::vector<Vec>& codebook) {
  Assignment result;
  result.indices.resize(weighted.size());
  std::uint64_t objective = 0;
  for (std::size_t index = 0; index < weighted.size(); ++index) {
    std::uint64_t distance = 0;
    result.indices[index] = Nearest(weighted[index].vector, codebook, &distance);
    if (distance != 0 &&
        weighted[index].count > std::numeric_limits<std::uint64_t>::max() / distance) {
      throw ArithmeticError("weighted Lloyd objective exceeds uint64");
    }
    const auto term = distance * weighted[index].count;
    if (term > std::numeric_limits<std::uint64_t>::max() - objective) {
      throw ArithmeticError("weighted Lloyd objective exceeds uint64");
    }
    objective += term;
  }
  result.objective = objective;
  return result;
}

std::int64_t CheckedWeightedValue(const std::int32_t value, const std::uint64_t count) {
  if (value >= 0) {
    if (value != 0 && count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
                                      static_cast<std::uint32_t>(value)) {
      throw ArithmeticError("weighted coordinate sum exceeds signed int64");
    }
    return static_cast<std::int64_t>(count * static_cast<std::uint32_t>(value));
  }
  const auto magnitude = static_cast<std::uint64_t>(-(static_cast<std::int64_t>(value) + 1)) + 1;
  const auto negative_limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1;
  if (count > negative_limit / magnitude) {
    throw ArithmeticError("weighted coordinate sum exceeds signed int64");
  }
  const auto product = count * magnitude;
  if (product == negative_limit) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return -static_cast<std::int64_t>(product);
}

std::int64_t CheckedAdd(const std::int64_t left, const std::int64_t right) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
    throw ArithmeticError("weighted coordinate sum exceeds signed int64");
  }
  return left + right;
}

std::int32_t RoundedMean(const std::int64_t sum, const std::uint64_t raw_count) {
  if (raw_count == 0) {
    throw std::logic_error("cannot round an empty weighted cluster");
  }
  if (raw_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw ArithmeticError("weighted-mean count exceeds signed int64");
  }
  const auto count = static_cast<std::int64_t>(raw_count);
  auto quotient = sum / count;
  auto remainder = sum % count;
  if (remainder < 0) {
    --quotient;
    remainder += count;
  }
  const auto rounded = remainder <= count / 2 ? quotient : quotient + 1;
  if (rounded < std::numeric_limits<std::int32_t>::min() ||
      rounded > std::numeric_limits<std::int32_t>::max()) {
    throw ArithmeticError("updated centroid is outside signed int32");
  }
  return static_cast<std::int32_t>(rounded);
}

std::vector<Vec> Update(const std::vector<WeightedVec>& weighted,
                        const std::vector<Vec>& codebook,
                        const std::vector<std::size_t>& assignments,
                        const Spec& spec) {
  if (assignments.size() != weighted.size()) {
    throw std::logic_error("assignment count disagrees with weighted-vector count");
  }
  std::vector<std::uint64_t> counts(codebook.size(), 0);
  std::vector<std::array<std::int64_t, 3>> sums(codebook.size());
  for (std::size_t index = 0; index < weighted.size(); ++index) {
    const auto cluster = assignments[index];
    if (cluster >= codebook.size() ||
        counts[cluster] > std::numeric_limits<std::uint64_t>::max() - weighted[index].count) {
      throw ArithmeticError("cluster count overflows uint64");
    }
    counts[cluster] += weighted[index].count;
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      const auto term = CheckedWeightedValue(weighted[index].vector.value[coordinate],
                                             weighted[index].count);
      sums[cluster][coordinate] = CheckedAdd(sums[cluster][coordinate], term);
    }
  }
  std::vector<Vec> updated = codebook;
  for (std::size_t cluster = 0; cluster < codebook.size(); ++cluster) {
    if (counts[cluster] == 0) {
      continue;
    }
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      updated[cluster].value[coordinate] = RoundedMean(sums[cluster][coordinate], counts[cluster]);
    }
  }
  std::sort(updated.begin(), updated.end());
  ValidateCentroids(updated, spec);
  return updated;
}

std::array<std::uint8_t, 32> StateHash(const std::vector<Vec>& codebook) {
  static constexpr char kDomain[] = "structsplat-comp011-vq-state-v1";
  if (codebook.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw ArithmeticError("codebook row count exceeds uint32 state hash grammar");
  }
  std::vector<std::uint8_t> bytes;
  bytes.reserve(sizeof(kDomain) + 4 + 12 * codebook.size());
  bytes.insert(bytes.end(), kDomain, kDomain + sizeof(kDomain));  // includes one NUL
  AppendU32Le(bytes, static_cast<std::uint32_t>(codebook.size()));
  for (const auto& vector : codebook) {
    for (const auto component : vector.value) {
      AppendU32Le(bytes, static_cast<std::uint32_t>(component));
    }
  }
  return Sha256(bytes);
}

struct FitResult {
  std::vector<Vec> codebook;
  std::vector<std::uint32_t> row_indices;
  std::vector<std::array<std::uint8_t, 32>> hashes;
  std::vector<std::uint64_t> objectives;
  std::uint32_t sweeps = 0;
  std::uint32_t status = 2;  // 0 fixed, 1 cycle, 2 cap
};

FitResult Fit(const std::int32_t* rows, const std::size_t row_count, const Spec& spec,
              const std::uint8_t start_id, const std::uint32_t maximum_sweeps) {
  ValidateSpec(spec, start_id);
  if (maximum_sweeps == 0 || maximum_sweeps > kMaximumSweeps) {
    throw std::invalid_argument("maximum scientific sweeps must lie in [1,100]");
  }
  const auto weighted = Collapse(rows, row_count, spec);
  auto codebook = Initialize(weighted, spec, start_id);
  auto assignment = Assign(weighted, codebook);
  FitResult result;
  result.hashes.push_back(StateHash(codebook));
  result.objectives.push_back(assignment.objective);
  std::vector<std::vector<Vec>> states = {codebook};

  for (std::uint32_t sweep = 1; sweep <= maximum_sweeps; ++sweep) {
    auto next = Update(weighted, codebook, assignment.indices, spec);
    auto next_assignment = Assign(weighted, next);
    if (next_assignment.objective > assignment.objective) {
      throw ArithmeticError("an exact Lloyd update increased its weighted objective");
    }
    result.hashes.push_back(StateHash(next));
    result.objectives.push_back(next_assignment.objective);
    result.sweeps = sweep;
    if (next == codebook) {
      result.status = 0;
      codebook = std::move(next);
      assignment = std::move(next_assignment);
      break;
    }
    bool cycle = false;
    for (const auto& old : states) {
      if (next == old) {
        cycle = true;
        break;
      }
    }
    codebook = std::move(next);
    assignment = std::move(next_assignment);
    if (cycle) {
      result.status = 1;
      break;
    }
    states.push_back(codebook);
  }

  result.codebook = codebook;
  result.row_indices.resize(row_count);
  for (std::size_t index = 0; index < row_count; ++index) {
    Vec vector;
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
      vector.value[coordinate] = rows[3 * index + coordinate];
    }
    const auto nearest = Nearest(vector, result.codebook);
    if (nearest > std::numeric_limits<std::uint32_t>::max()) {
      throw ArithmeticError("materialized index exceeds uint32");
    }
    result.row_indices[index] = static_cast<std::uint32_t>(nearest);
  }
  return result;
}

std::pair<std::vector<Vec>, std::array<std::uint8_t, 32>> ForceSweeps(
    const std::int32_t* rows, const std::size_t row_count, const Spec& spec,
    const std::uint8_t start_id, const std::uint32_t forced_sweeps) {
  ValidateSpec(spec, start_id);
  if (forced_sweeps == 0) {
    throw std::invalid_argument("forced sweep count must be positive");
  }
  const auto weighted = Collapse(rows, row_count, spec);
  auto codebook = Initialize(weighted, spec, start_id);
  for (std::uint32_t sweep = 0; sweep < forced_sweeps; ++sweep) {
    const auto assignment = Assign(weighted, codebook);
    codebook = Update(weighted, codebook, assignment.indices, spec);
  }
  return {codebook, StateHash(codebook)};
}

template <typename Callable>
int Guard(Callable&& callable) {
  try {
    callable();
    g_last_error.clear();
    return kSuccess;
  } catch (const std::invalid_argument& error) {
    g_last_error = error.what();
    return kInvalidArgument;
  } catch (const ArithmeticError& error) {
    g_last_error = error.what();
    return kArithmeticError;
  } catch (const BufferTooSmall& error) {
    g_last_error = error.what();
    return kBufferTooSmall;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return kInternalError;
  } catch (...) {
    g_last_error = "unknown native exception";
    return kInternalError;
  }
}

}  // namespace

extern "C" {

const char* ssp2v_lloyd_last_error() { return g_last_error.c_str(); }

int ssp2v_lloyd_fit(
    const std::int32_t* rows, const std::size_t row_count, const std::size_t codebook_size,
    const std::uint8_t family_id, const std::uint8_t matched_stage_count,
    const std::uint16_t base_k, const std::uint8_t actual_stage_index,
    const std::uint8_t start_id, const std::uint32_t maximum_sweeps,
    std::int32_t* output_codebook, std::uint32_t* output_indices,
    std::uint8_t* output_trajectory_hashes, std::uint64_t* output_trajectory_objectives,
    const std::size_t trajectory_capacity, std::size_t* output_trajectory_count,
    std::uint32_t* output_update_sweeps, std::uint32_t* output_convergence_status,
    std::uint64_t* output_terminal_sse) {
  if (output_codebook == nullptr || output_indices == nullptr ||
      output_trajectory_hashes == nullptr || output_trajectory_objectives == nullptr ||
      output_trajectory_count == nullptr || output_update_sweeps == nullptr ||
      output_convergence_status == nullptr || output_terminal_sse == nullptr) {
    g_last_error = "a required output pointer is null";
    return kInvalidArgument;
  }
  return Guard([&]() {
    const Spec spec{family_id, matched_stage_count, base_k, actual_stage_index, codebook_size};
    const auto result = Fit(rows, row_count, spec, start_id, maximum_sweeps);
    if (trajectory_capacity < result.hashes.size()) {
      throw BufferTooSmall("trajectory output capacity is too small");
    }
    for (std::size_t index = 0; index < result.codebook.size(); ++index) {
      for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
        output_codebook[3 * index + coordinate] = result.codebook[index].value[coordinate];
      }
    }
    std::memcpy(output_indices, result.row_indices.data(),
                result.row_indices.size() * sizeof(std::uint32_t));
    for (std::size_t index = 0; index < result.hashes.size(); ++index) {
      std::memcpy(output_trajectory_hashes + 32 * index, result.hashes[index].data(), 32);
      output_trajectory_objectives[index] = result.objectives[index];
    }
    *output_trajectory_count = result.hashes.size();
    *output_update_sweeps = result.sweeps;
    *output_convergence_status = result.status;
    *output_terminal_sse = result.objectives.back();
  });
}

int ssp2v_lloyd_force_sweeps(
    const std::int32_t* rows, const std::size_t row_count, const std::size_t codebook_size,
    const std::uint8_t family_id, const std::uint8_t matched_stage_count,
    const std::uint16_t base_k, const std::uint8_t actual_stage_index,
    const std::uint8_t start_id, const std::uint32_t forced_sweeps,
    std::int32_t* output_codebook, std::uint8_t* output_hash) {
  if (output_codebook == nullptr || output_hash == nullptr) {
    g_last_error = "a required forced-sweep output pointer is null";
    return kInvalidArgument;
  }
  return Guard([&]() {
    const Spec spec{family_id, matched_stage_count, base_k, actual_stage_index, codebook_size};
    const auto result = ForceSweeps(rows, row_count, spec, start_id, forced_sweeps);
    for (std::size_t index = 0; index < result.first.size(); ++index) {
      for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
        output_codebook[3 * index + coordinate] = result.first[index].value[coordinate];
      }
    }
    std::memcpy(output_hash, result.second.data(), result.second.size());
  });
}

}  // extern "C"
