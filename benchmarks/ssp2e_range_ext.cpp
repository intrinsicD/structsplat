#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kFull = std::uint64_t{1} << 32;
constexpr std::uint32_t kMask = 0xFFFFFFFFu;
constexpr std::uint32_t kHalf = std::uint32_t{1} << 31;
constexpr std::uint32_t kQ1 = std::uint32_t{1} << 30;
constexpr std::uint32_t kQ3 = std::uint32_t{3} << 30;
constexpr std::uint32_t kTotal = 32768;

constexpr int kSuccess = 0;
constexpr int kInvalidArgument = 1;
constexpr int kBufferTooSmall = 2;
constexpr int kInvalidModel = 3;
constexpr int kMalformedPayload = 4;
constexpr int kInternalError = 5;

thread_local std::string g_last_error;

class BufferTooSmall final : public std::exception {
 public:
  const char* what() const noexcept override { return "output buffer is too small"; }
};

class BitWriter {
 public:
  void Write(const std::uint32_t bit) {
    if (bit > 1) {
      throw std::logic_error("attempted to emit a non-bit");
    }
    byte_ = static_cast<std::uint8_t>((byte_ << 1) | bit);
    ++used_;
    if (used_ == 8) {
      bytes_.push_back(byte_);
      byte_ = 0;
      used_ = 0;
    }
  }

  void WriteRepeated(const std::uint32_t bit, std::size_t repetitions) {
    while (repetitions != 0) {
      Write(bit);
      --repetitions;
    }
  }

  std::vector<std::uint8_t> Finish() {
    if (used_ != 0) {
      bytes_.push_back(static_cast<std::uint8_t>(byte_ << (8 - used_)));
      byte_ = 0;
      used_ = 0;
    }
    return bytes_;
  }

 private:
  std::vector<std::uint8_t> bytes_;
  std::uint8_t byte_ = 0;
  int used_ = 0;
};

class CountSink {
 public:
  void Write(const std::uint32_t bit) { WriteRepeated(bit, 1); }

  void WriteRepeated(const std::uint32_t bit, const std::size_t repetitions) {
    if (bit > 1) {
      throw std::logic_error("attempted to emit a non-bit");
    }
    if (repetitions > std::numeric_limits<std::size_t>::max() - bit_count_) {
      throw std::overflow_error("arithmetic output bit count overflows size_t");
    }
    bit_count_ += repetitions;
  }

  std::size_t Finish() const {
    return bit_count_ / 8 + static_cast<std::size_t>(bit_count_ % 8 != 0);
  }

 private:
  std::size_t bit_count_ = 0;
};

class BitReader {
 public:
  BitReader(const std::uint8_t* payload, const std::size_t size)
      : payload_(payload), size_(size) {}

  std::uint32_t Read() {
    const auto position = position_++;
    if (position / 8 >= size_) {
      return 0;
    }
    return (payload_[position / 8] >> (7 - position % 8)) & 1;
  }

 private:
  const std::uint8_t* payload_;
  std::size_t size_;
  std::size_t position_ = 0;
};

void ValidateAlphabet(const std::size_t alphabet) {
  if (alphabet < 2 || alphabet > kTotal) {
    throw std::invalid_argument("CDF alphabet must lie in [2, 32768]");
  }
}

void ValidateCount(const std::size_t count) {
  // A nonempty 32-bit arithmetic interval can be renormalized at most 32
  // times per symbol.  Two further bits terminate the stream.
  constexpr auto kRenormalizationsPerSymbol = std::size_t{32};
  constexpr auto kTerminationBits = std::size_t{2};
  if (count >
      (std::numeric_limits<std::size_t>::max() - kTerminationBits) /
          kRenormalizationsPerSymbol) {
    throw std::overflow_error("symbol count exceeds the exact bit-counter range");
  }
}

void ValidateRow(const std::uint32_t* row, const std::size_t alphabet) {
  if (row[0] != 0 || row[alphabet] != kTotal) {
    throw std::invalid_argument("CDF row has an invalid endpoint");
  }
  for (std::size_t index = 0; index < alphabet; ++index) {
    if (row[index] >= row[index + 1]) {
      throw std::invalid_argument("CDF row is not strictly increasing");
    }
  }
}

void ValidateRows(const std::uint32_t* cdfs, const std::size_t count,
                  const std::size_t alphabet) {
  ValidateAlphabet(alphabet);
  ValidateCount(count);
  if (count != 0 && cdfs == nullptr) {
    throw std::invalid_argument("CDF pointer is null");
  }
  const auto stride = alphabet + 1;
  if (count > std::numeric_limits<std::size_t>::max() / stride) {
    throw std::overflow_error("CDF matrix element count overflows size_t");
  }
  for (std::size_t row_index = 0; row_index < count; ++row_index) {
    ValidateRow(cdfs + row_index * stride, alphabet);
  }
}

void ValidateRepeatedRow(const std::uint32_t* cdf, const std::size_t count,
                         const std::size_t alphabet) {
  ValidateAlphabet(alphabet);
  ValidateCount(count);
  if (cdf == nullptr) {
    throw std::invalid_argument("repeated CDF pointer is null");
  }
  ValidateRow(cdf, alphabet);
}

template <typename Writer>
void EmitWithPending(Writer& writer, const std::uint32_t bit, std::size_t& pending) {
  writer.Write(bit);
  writer.WriteRepeated(1 - bit, pending);
  pending = 0;
}

void IncrementPending(std::size_t& pending) {
  if (pending == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("arithmetic pending-bit count overflows size_t");
  }
  ++pending;
}

class MatrixRows {
 public:
  MatrixRows(const std::uint32_t* cdfs, const std::size_t alphabet)
      : cdfs_(cdfs), stride_(alphabet + 1) {}

  const std::uint32_t* operator()(const std::size_t index) const {
    return cdfs_ + index * stride_;
  }

 private:
  const std::uint32_t* cdfs_;
  std::size_t stride_;
};

class RepeatedRow {
 public:
  explicit RepeatedRow(const std::uint32_t* cdf) : cdf_(cdf) {}

  const std::uint32_t* operator()(const std::size_t) const { return cdf_; }

 private:
  const std::uint32_t* cdf_;
};

template <typename Writer, typename RowAccessor>
void EncodeCore(const std::uint32_t* symbols, const std::size_t count,
                const std::size_t alphabet, const RowAccessor& row_at,
                Writer& writer) {
  std::uint32_t low = 0;
  std::uint32_t high = kMask;
  std::size_t pending = 0;

  for (std::size_t row_index = 0; row_index < count; ++row_index) {
    const auto symbol = symbols[row_index];
    if (symbol >= alphabet) {
      throw std::invalid_argument("symbol is outside the CDF alphabet");
    }
    const auto* row = row_at(row_index);
    const std::uint64_t range =
        static_cast<std::uint64_t>(high) - static_cast<std::uint64_t>(low) + 1;
    const std::uint64_t unit = range / kTotal;
    if (unit == 0) {
      throw std::runtime_error("arithmetic interval cannot represent the frequency total");
    }
    const std::uint64_t child_low =
        static_cast<std::uint64_t>(low) + unit * row[symbol];
    const std::uint64_t child_high = child_low + unit * (row[symbol + 1] - row[symbol]) - 1;
    if (child_high > kMask || child_low > child_high) {
      throw std::runtime_error("arithmetic child interval overflow");
    }
    low = static_cast<std::uint32_t>(child_low);
    high = static_cast<std::uint32_t>(child_high);

    for (;;) {
      if (high < kHalf) {
        EmitWithPending(writer, 0, pending);
      } else if (low >= kHalf) {
        EmitWithPending(writer, 1, pending);
        low -= kHalf;
        high -= kHalf;
      } else if (low >= kQ1 && high < kQ3) {
        IncrementPending(pending);
        low -= kQ1;
        high -= kQ1;
      } else {
        break;
      }
      low = static_cast<std::uint32_t>((static_cast<std::uint64_t>(low) << 1) & kMask);
      high = static_cast<std::uint32_t>(
          ((static_cast<std::uint64_t>(high) << 1) & kMask) | 1);
    }
  }

  IncrementPending(pending);
  if (low < kQ1) {
    EmitWithPending(writer, 0, pending);
  } else {
    EmitWithPending(writer, 1, pending);
  }
}

std::vector<std::uint8_t> EncodeImpl(const std::uint32_t* symbols,
                                     const std::uint32_t* cdfs,
                                     const std::size_t count,
                                     const std::size_t alphabet) {
  ValidateRows(cdfs, count, alphabet);
  if (count != 0 && symbols == nullptr) {
    throw std::invalid_argument("symbol pointer is null");
  }
  BitWriter writer;
  EncodeCore(symbols, count, alphabet, MatrixRows(cdfs, alphabet), writer);
  return writer.Finish();
}

std::size_t SizeRepeatedImpl(const std::uint32_t* symbols,
                             const std::uint32_t* cdf,
                             const std::size_t count,
                             const std::size_t alphabet) {
  ValidateRepeatedRow(cdf, count, alphabet);
  if (count != 0 && symbols == nullptr) {
    throw std::invalid_argument("symbol pointer is null");
  }
  CountSink sink;
  EncodeCore(symbols, count, alphabet, RepeatedRow(cdf), sink);
  return sink.Finish();
}

std::vector<std::uint32_t> DecodeImpl(const std::uint8_t* payload,
                                      const std::size_t payload_size,
                                      const std::uint32_t* cdfs,
                                      const std::size_t count,
                                      const std::size_t alphabet) {
  if (payload == nullptr || payload_size == 0) {
    throw std::domain_error("canonical arithmetic payload cannot be empty");
  }
  ValidateRows(cdfs, count, alphabet);
  BitReader reader(payload, payload_size);
  std::uint32_t low = 0;
  std::uint32_t high = kMask;
  std::uint32_t code = 0;
  for (int bit = 0; bit < 32; ++bit) {
    code = static_cast<std::uint32_t>((static_cast<std::uint64_t>(code) << 1) | reader.Read());
  }

  std::vector<std::uint32_t> symbols(count);
  const auto stride = alphabet + 1;
  for (std::size_t row_index = 0; row_index < count; ++row_index) {
    if (code < low || code > high) {
      throw std::domain_error("arithmetic code lies outside the current interval");
    }
    const std::uint64_t range =
        static_cast<std::uint64_t>(high) - static_cast<std::uint64_t>(low) + 1;
    const std::uint64_t unit = range / kTotal;
    if (unit == 0) {
      throw std::domain_error("arithmetic interval cannot represent the frequency total");
    }
    const std::uint64_t scaled = (static_cast<std::uint64_t>(code) - low) / unit;
    if (scaled >= kTotal) {
      throw std::domain_error("arithmetic code entered the unassigned interval tail");
    }
    const auto* row = cdfs + row_index * stride;
    const auto* upper = std::upper_bound(row, row + stride, static_cast<std::uint32_t>(scaled));
    if (upper == row || upper == row + stride) {
      throw std::domain_error("arithmetic code does not select a CDF interval");
    }
    const auto symbol = static_cast<std::size_t>(upper - row - 1);
    if (symbol >= alphabet || !(row[symbol] <= scaled && scaled < row[symbol + 1])) {
      throw std::domain_error("arithmetic code does not select a positive CDF interval");
    }
    symbols[row_index] = static_cast<std::uint32_t>(symbol);
    const std::uint64_t child_low =
        static_cast<std::uint64_t>(low) + unit * row[symbol];
    const std::uint64_t child_high =
        static_cast<std::uint64_t>(low) + unit * row[symbol + 1] - 1;
    low = static_cast<std::uint32_t>(child_low);
    high = static_cast<std::uint32_t>(child_high);

    for (;;) {
      if (high < kHalf) {
        // No translation is needed for E1.
      } else if (low >= kHalf) {
        low -= kHalf;
        high -= kHalf;
        code -= kHalf;
      } else if (low >= kQ1 && high < kQ3) {
        low -= kQ1;
        high -= kQ1;
        code -= kQ1;
      } else {
        break;
      }
      low = static_cast<std::uint32_t>((static_cast<std::uint64_t>(low) << 1) & kMask);
      high = static_cast<std::uint32_t>(
          ((static_cast<std::uint64_t>(high) << 1) & kMask) | 1);
      code = static_cast<std::uint32_t>(
          ((static_cast<std::uint64_t>(code) << 1) & kMask) | reader.Read());
    }
  }

  const auto canonical = EncodeImpl(symbols.data(), cdfs, count, alphabet);
  if (canonical.size() != payload_size ||
      std::memcmp(canonical.data(), payload, payload_size) != 0) {
    throw std::domain_error("payload is not the canonical encoding for count and model");
  }
  return symbols;
}

template <typename Callable>
int Guard(const int error_code, Callable&& callable) {
  try {
    callable();
    g_last_error.clear();
    return kSuccess;
  } catch (const std::invalid_argument& error) {
    g_last_error = error.what();
    return kInvalidModel;
  } catch (const std::domain_error& error) {
    g_last_error = error.what();
    return kMalformedPayload;
  } catch (const BufferTooSmall& error) {
    g_last_error = error.what();
    return kBufferTooSmall;
  } catch (const std::overflow_error& error) {
    g_last_error = error.what();
    return kInvalidArgument;
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return error_code;
  } catch (...) {
    g_last_error = "unknown native exception";
    return kInternalError;
  }
}

}  // namespace

extern "C" {

const char* ssp2e_last_error() { return g_last_error.c_str(); }

int ssp2e_encode(const std::uint32_t* symbols, const std::uint32_t* cdfs,
                 const std::size_t count, const std::size_t alphabet,
                 std::uint8_t* output, const std::size_t output_capacity,
                 std::size_t* output_size) {
  if (output_size == nullptr) {
    g_last_error = "output-size pointer is null";
    return kInvalidArgument;
  }
  *output_size = 0;
  return Guard(kInternalError, [&]() {
    const auto encoded = EncodeImpl(symbols, cdfs, count, alphabet);
    *output_size = encoded.size();
    if (output == nullptr || output_capacity < encoded.size()) {
      throw BufferTooSmall();
    }
    std::memcpy(output, encoded.data(), encoded.size());
  });
}

int ssp2e_size_repeated(const std::uint32_t* symbols,
                        const std::uint32_t* cdf,
                        const std::size_t count,
                        const std::size_t alphabet,
                        std::size_t* output_size) {
  if (output_size == nullptr) {
    g_last_error = "output-size pointer is null";
    return kInvalidArgument;
  }
  *output_size = 0;
  if (cdf == nullptr) {
    g_last_error = "repeated CDF pointer is null";
    return kInvalidArgument;
  }
  if (count != 0 && symbols == nullptr) {
    g_last_error = "symbol pointer is null";
    return kInvalidArgument;
  }
  return Guard(kInternalError, [&]() {
    *output_size = SizeRepeatedImpl(symbols, cdf, count, alphabet);
  });
}

int ssp2e_decode(const std::uint8_t* payload, const std::size_t payload_size,
                 const std::uint32_t* cdfs, const std::size_t count,
                 const std::size_t alphabet, std::uint32_t* symbols) {
  if (count != 0 && symbols == nullptr) {
    g_last_error = "output-symbol pointer is null";
    return kInvalidArgument;
  }
  return Guard(kInternalError, [&]() {
    const auto decoded = DecodeImpl(payload, payload_size, cdfs, count, alphabet);
    if (count != 0) {
      std::memcpy(symbols, decoded.data(), count * sizeof(std::uint32_t));
    }
  });
}

}  // extern "C"
