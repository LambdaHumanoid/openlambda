// SPDX-License-Identifier: Apache-2.0
// OpenLambda optional terminal-runtime adapter. No robot start/enable commands.
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <zmq.h>

class OpenLambdaRuntime {
 public:
  OpenLambdaRuntime() {
    const char* enabled = std::getenv("OPENLAMBDA_RUNTIME");
    if (!enabled || std::strcmp(enabled, "1") != 0) return;
    context_ = zmq_ctx_new();
    socket_ = context_ ? zmq_socket(context_, ZMQ_PAIR) : nullptr;
    if (!socket_) {
      if (context_) zmq_ctx_term(context_);
      throw std::runtime_error("OpenLambda ZMQ allocation failed");
    }
    int zero = 0, hwm = 4;
    if (zmq_setsockopt(socket_, ZMQ_LINGER, &zero, sizeof(zero)) != 0 ||
        zmq_setsockopt(socket_, ZMQ_SNDHWM, &hwm, sizeof(hwm)) != 0 ||
        zmq_setsockopt(socket_, ZMQ_RCVHWM, &hwm, sizeof(hwm)) != 0 ||
        zmq_bind(socket_, "tcp://127.0.0.1:5686") != 0) {
      zmq_close(socket_);
      zmq_ctx_term(context_);
      throw std::runtime_error("OpenLambda backend setup failed or port 5686 already owned");
    }
  }
  ~OpenLambdaRuntime() {
    if (socket_) zmq_close(socket_);
    if (context_) zmq_ctx_term(context_);
  }
  OpenLambdaRuntime(const OpenLambdaRuntime&) = delete;
  OpenLambdaRuntime& operator=(const OpenLambdaRuntime&) = delete;

  bool Enabled() const { return socket_ != nullptr; }

  // WAIT_FOR_CONTROL is explicitly different from successful CONTROL feedback.
  void Waiting(const std::vector<double>& token, bool external) {
    compatible_ = false;
    waiting_ = external;
    owner_ = 0;
    latched_ = false;
    watchdog_ = false;
    After(token);
  }

  // Call inside the motion lock, AFTER input gathering and BEFORE observation
  // construction. Local encoder/planner operation is never overridden.
  void Before(std::vector<double>& token, bool external) {
    if (!socket_) return;
    waiting_ = false;
    compatible_ = external && token.size() == 64;
    for (double value : token)
      compatible_ = compatible_ && std::isfinite(value) && std::abs(value) <= 1.25;
    const auto now = Now();
    std::array<unsigned char, 290> packet{};
    for (int i = 0; i < 4; ++i) {
      int size = zmq_recv(socket_, packet.data(), packet.size(), ZMQ_DONTWAIT);
      if (size < 0) break;
      if (!compatible_ || size != 289 || std::memcmp(packet.data(), "OLCMD001", 8)) continue;
      const unsigned char kind = packet[8];
      const auto owner = Read64(packet.data() + 9);
      const auto sequence = Read64(packet.data() + 17);
      const auto sent = Read64(packet.data() + 25);
      if (!owner || sent > now || now - sent > 250000000ULL) continue;
      std::array<float, 64> values{};
      std::memcpy(values.data(), packet.data() + 33, 256);
      bool valid = true;
      for (float value : values) valid = valid && std::isfinite(value) && std::abs(value) <= 1.25f;
      if (!valid) continue;
      if (kind == 1 && owner_ == 0) {
        // A Hold request may only capture the token already used by CONTROL.
        bool same = true;
        for (int j = 0; j < 64; ++j) same = same && std::abs(token[j] - values[j]) <= 1e-5;
        if (!same) continue;
        for (int j = 0; j < 64; ++j) hold_[j] = static_cast<float>(token[j]);
        target_ = hold_;
        owner_ = owner;
        sequence_ = sequence;
        latched_ = true;
        last_seen_ = now;
        last_policy_ = 0;
        continue;
      }
      if (owner != owner_ || sequence <= sequence_) continue;
      if (kind == 2) { target_ = values; last_policy_ = now; }
      else if (kind == 3) { /* owner heartbeat; not a fresh policy target */ }
      else if (kind == 4) { target_ = hold_; owner_ = 0; last_policy_ = 0; }
      else if (kind == 5) { target_ = hold_; last_policy_ = 0; }
      else continue;
      sequence_ = sequence;
      last_seen_ = now;
    }
    watchdog_ = latched_ && (now - last_seen_ > 250000000ULL);
    if (watchdog_) { owner_ = 0; last_policy_ = 0; }
    if (!last_policy_ || now - last_policy_ > 250000000ULL || watchdog_) target_ = hold_;
    if (compatible_ && latched_)
      for (int j = 0; j < 64; ++j) token[j] = target_[j];
  }

  // Only call after successful policy inference. This is applied-controller
  // feedback, not a telemetry-derived assertion that an idle robot is in Hold.
  void After(const std::vector<double>& token) {
    if (!socket_ || token.size() != 64) return;
    std::array<unsigned char, 289> packet{};
    std::memcpy(packet.data(), "OLACK001", 8);
    Write64(packet.data() + 8, Now());
    Write64(packet.data() + 16, owner_);
    Write64(packet.data() + 24, sequence_);
    packet[32] = (compatible_ ? 1 : 0) | (owner_ ? 2 : 0) |
                 (latched_ && target_ == hold_ ? 4 : 0) | (watchdog_ ? 8 : 0) |
                 (waiting_ ? 16 : 0);
    for (int j = 0; j < 64; ++j) {
      float value = static_cast<float>(token[j]);
      std::memcpy(packet.data() + 33 + j * 4, &value, 4);
    }
    zmq_send(socket_, packet.data(), packet.size(), ZMQ_DONTWAIT);
  }

 private:
  static std::uint64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  static std::uint64_t Read64(const unsigned char* p) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= std::uint64_t(p[i]) << (8 * i);
    return value;
  }
  static void Write64(unsigned char* p, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(value >> (8 * i));
  }
  void* context_ = nullptr;
  void* socket_ = nullptr;
  std::uint64_t owner_ = 0, sequence_ = 0, last_seen_ = 0, last_policy_ = 0;
  bool compatible_ = false, latched_ = false, watchdog_ = false, waiting_ = false;
  std::array<float, 64> hold_{}, target_{};
};
