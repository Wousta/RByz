#pragma once
#include <hiredis/hiredis.h>
#include <iostream>
#include <string>

inline void RedisCnct(redisContext *&context, std::string ipv4_addr, int port) {
  context = redisConnect(ipv4_addr.c_str(), port);
  if (context == nullptr) {
    std::cerr << "failed to connect to Redis\n";
  }
}
inline void RedisDcnct(redisContext *&context) { redisFree(context); }
inline void SET(std::string key, std::string value, redisContext *&context) {
  std::string cmd("SET ");
  cmd = cmd + key;
  cmd = cmd + " ";
  cmd = cmd + value;
  const char *setCommand = cmd.c_str();
  redisReply *reply = (redisReply *)redisCommand(context, setCommand);

  if (reply == nullptr) {
    std::cerr << "Error executing SET command: " << context->errstr
              << std::endl;
    redisFree(context);
    return;
  }
  freeReplyObject(reply);
}
inline std::string GET(std::string key, redisContext *&context) {
  std::string cmd("GET ");
  cmd = cmd + key;
  const char *setCommand = cmd.c_str();
  redisReply *reply = (redisReply *)redisCommand(context, setCommand);

  if (reply == nullptr) {
    std::cerr << "Error executing SET command: " << context->errstr
              << std::endl;
    redisFree(context);
    freeReplyObject(reply);
    return "";
  }
  if (reply->type == REDIS_REPLY_STRING) {
    if (reply->str != nullptr && reply->len > 0) {
      std::string result(reply->str, reply->len);
      freeReplyObject(reply);
      return result;
    }
  }
  freeReplyObject(reply);
  return "";
}