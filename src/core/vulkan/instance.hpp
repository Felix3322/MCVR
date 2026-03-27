#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance : public SharedObject<Instance> {
  public:
    Instance();
    ~Instance();

    VkInstance &vkInstance();
    bool isDlssInstanceExtensionsCompatible() const;
    bool isDlssFrameGenerationInstanceExtensionsCompatible() const;
    bool isXessInstanceExtensionsCompatible() const;

  private:
    VkInstance instance_;
    bool dlssInstanceExtensionsCompatible_ = false;
    bool dlssFrameGenerationInstanceExtensionsCompatible_ = false;
    bool xessInstanceExtensionsCompatible_ = false;
    // VkDebugReportCallbackEXT callback_ = VK_NULL_HANDLE;
};
} // namespace vk
