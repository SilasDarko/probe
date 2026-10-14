// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/json_writer.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace probe {

void JsonWriter::before_value() {
    if (after_key_) {
        after_key_ = false;  // the comma, if any, was written before the key
        return;
    }
    if (!container_is_empty_.empty()) {
        if (!container_is_empty_.back()) out_.push_back(',');
        container_is_empty_.back() = false;
    }
}

void JsonWriter::write_escaped(std::string_view s) {
    out_.push_back('"');
    for (const char c : s) {
        switch (c) {
            case '"':
                out_ += "\\\"";
                break;
            case '\\':
                out_ += "\\\\";
                break;
            case '\n':
                out_ += "\\n";
                break;
            case '\t':
                out_ += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
                    out_ += buf;
                } else {
                    out_.push_back(c);
                }
        }
    }
    out_.push_back('"');
}

JsonWriter& JsonWriter::begin_object() {
    before_value();
    out_.push_back('{');
    container_is_empty_.push_back(true);
    return *this;
}

JsonWriter& JsonWriter::end_object() {
    container_is_empty_.pop_back();
    out_.push_back('}');
    return *this;
}

JsonWriter& JsonWriter::begin_array() {
    before_value();
    out_.push_back('[');
    container_is_empty_.push_back(true);
    return *this;
}

JsonWriter& JsonWriter::end_array() {
    container_is_empty_.pop_back();
    out_.push_back(']');
    return *this;
}

JsonWriter& JsonWriter::key(std::string_view name) {
    before_value();
    write_escaped(name);
    out_.push_back(':');
    after_key_ = true;
    return *this;
}

JsonWriter& JsonWriter::string(std::string_view value) {
    before_value();
    write_escaped(value);
    return *this;
}

JsonWriter& JsonWriter::integer(long long value) {
    before_value();
    out_ += std::to_string(value);
    return *this;
}

JsonWriter& JsonWriter::uinteger(unsigned long long value) {
    before_value();
    out_ += std::to_string(value);
    return *this;
}

JsonWriter& JsonWriter::number(double value) {
    before_value();
    if (!std::isfinite(value)) {
        out_ += "null";
        return *this;
    }
    // Shortest round-trip form, e.g. "1.5", "0.1", "1e-07". Every output of
    // to_chars for a finite double is also a valid JSON number.
    char buf[32];
    const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, value);
    out_.append(buf, end);
    return *this;
}

JsonWriter& JsonWriter::boolean(bool value) {
    before_value();
    out_ += value ? "true" : "false";
    return *this;
}

JsonWriter& JsonWriter::null() {
    before_value();
    out_ += "null";
    return *this;
}

}  // namespace probe
