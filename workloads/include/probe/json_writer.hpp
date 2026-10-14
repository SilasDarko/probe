// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace probe {

// A minimal streaming JSON writer. It handles commas, string escaping and
// number formatting, which is all probe-workloads needs, so the project does
// not pull in a third-party JSON library. It does not validate call order
// (for example, a key outside an object); callers are internal and tested.
//
//   JsonWriter w;
//   w.begin_object().key("n").integer(3).key("ok").boolean(true).end_object();
//   w.str()  ->  {"n":3,"ok":true}
class JsonWriter {
   public:
    JsonWriter& begin_object();
    JsonWriter& end_object();
    JsonWriter& begin_array();
    JsonWriter& end_array();
    JsonWriter& key(std::string_view name);

    JsonWriter& string(std::string_view value);
    JsonWriter& integer(long long value);
    JsonWriter& uinteger(unsigned long long value);
    // Non-finite values (NaN, infinity) are not valid JSON and are written as
    // null. Finite values use 9 significant digits, enough to round-trip any
    // float exactly.
    JsonWriter& number(double value);
    JsonWriter& boolean(bool value);
    JsonWriter& null();

    const std::string& str() const { return out_; }

   private:
    void before_value();
    void write_escaped(std::string_view s);

    std::string out_;
    std::vector<bool> container_is_empty_;  // one entry per open object/array
    bool after_key_ = false;
};

}  // namespace probe
