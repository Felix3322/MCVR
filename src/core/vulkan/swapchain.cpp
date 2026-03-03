#include "core/vulkan/swapchain.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/window.hpp"

#include "core/render/renderer.hpp"
#include "core/render/streamline_context.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

std::ostream &swapchainCout() {
    return std::cout << "[Swapchain] ";
}

std::ostream &swapchainCerr() {
    return std::cerr << "[Swapchain] ";
}

vk::Swapchain::Swapchain(std::shared_ptr<PhysicalDevice> physicalDevice,
                         std::shared_ptr<Device> device,
                         std::shared_ptr<Window> window)
    : physicalDevice_(physicalDevice), device_(device), window_(window) {
    reconstruct();
}

VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // can choose any format
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
#ifdef DEBUG
        std::cout << "selected surface format: " << VK_FORMAT_R8G8B8A8_UNORM
                  << " color space: " << VK_COLORSPACE_SRGB_NONLINEAR_KHR << std::endl;
#endif
        return {VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    }

    if (Renderer::options.hdrOutput) {
        for (const auto &availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                availableFormat.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
#ifdef DEBUG
                std::cout << "selected HDR surface format: " << availableFormat.format
                          << " color space: " << availableFormat.colorSpace << std::endl;
#endif
                return availableFormat;
            }
        }
        swapchainCerr() << "requested HDR output, but scRGB swapchain format is unavailable; falling back to SDR"
                        << std::endl;
    }

    auto formatRank = [](VkFormat format) {
        switch (format) {
            case VK_FORMAT_R8G8B8A8_UNORM: return 0;
            case VK_FORMAT_B8G8R8A8_UNORM: return 1;
            case VK_FORMAT_R8G8B8A8_SRGB: return 2;
            case VK_FORMAT_B8G8R8A8_SRGB: return 3;
            default: return 4;
        }
    };

    auto selectedFormat = std::min_element(
        availableFormats.begin(), availableFormats.end(),
        [&formatRank](const auto &lhs, const auto &rhs) { return formatRank(lhs.format) < formatRank(rhs.format); });

    if (selectedFormat->format == VK_FORMAT_R8G8B8A8_SRGB || selectedFormat->format == VK_FORMAT_B8G8R8A8_SRGB) {
        swapchainCerr() << "warning: selected SRGB surface format (" << selectedFormat->format
                        << "), UNORM format unavailable, this may cause color space error" << std::endl;
    }

#ifdef DEBUG
    std::cout << "selected surface format: " << selectedFormat->format << " color space: " << selectedFormat->colorSpace
              << std::endl;
#endif
    return *selectedFormat;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities, uint32_t width, uint32_t height) {
    if (surfaceCapabilities.currentExtent.width == -1) {
        VkExtent2D swapChainExtent = {};

        swapChainExtent.width = std::min(std::max(width, surfaceCapabilities.minImageExtent.width),
                                         surfaceCapabilities.maxImageExtent.width);
        swapChainExtent.height = std::min(std::max(height, surfaceCapabilities.minImageExtent.height),
                                          surfaceCapabilities.maxImageExtent.height);
        return swapChainExtent;
    } else {
        return surfaceCapabilities.currentExtent;
    }
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR> presentModes) {
    bool fgSwapchainActive = Renderer::options.dlssFrameGenerationActive;

    auto supportsMode = [&presentModes](VkPresentModeKHR mode) {
        return std::find(presentModes.begin(), presentModes.end(), mode) != presentModes.end();
    };

    if (fgSwapchainActive) {
        if (supportsMode(VK_PRESENT_MODE_MAILBOX_KHR)) {
            swapchainCout() << "DLSS Frame Generation using MAILBOX present mode to avoid halving render cadence."
                            << std::endl;
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
        if (supportsMode(VK_PRESENT_MODE_IMMEDIATE_KHR)) {
            swapchainCout() << "DLSS Frame Generation falling back to IMMEDIATE present mode; MAILBOX unavailable."
                            << std::endl;
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
        }

        swapchainCerr() << "DLSS Frame Generation is limited to FIFO present mode on this device; "
                           "interpolated frames may be skipped to protect base FPS."
                        << std::endl;
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    if (Renderer::options.vsync) {
        // When Streamline Reflex is available, prefer MAILBOX over FIFO.
        // FIFO blocks the CPU thread at vkQueuePresentKHR until vblank, which:
        //   1. Prevents slReflexSleep from controlling frame pacing
        //   2. Adds unavoidable latency that Reflex can't compensate for
        //   3. Defeats GSYNC's variable refresh rate (display always waits for vblank)
        // MAILBOX doesn't block at present (still no tearing — display flips at vblank)
        // so Reflex can pace frames via sleep and GSYNC can adapt the refresh rate.
        if (StreamlineContext::isReflexAvailable()) {
            for (const auto &presentMode : presentModes) {
                if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    swapchainCout() << "vsync+Reflex: using MAILBOX for Reflex frame pacing" << std::endl;
                    return presentMode;
                }
            }
            swapchainCout() << "vsync+Reflex: MAILBOX unavailable, falling back to FIFO" << std::endl;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    if (supportsMode(VK_PRESENT_MODE_IMMEDIATE_KHR)) { return VK_PRESENT_MODE_IMMEDIATE_KHR; }

    // If immediate is unavailable, fall back to FIFO (guaranteed to be available)
    return VK_PRESENT_MODE_FIFO_KHR;
}

void vk::Swapchain::reconstruct() {
    // Find surface capabilities
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &surfaceCapabilities) != VK_SUCCESS) {
        swapchainCerr() << "failed to acquire presentation surface capabilities" << std::endl;
        exit(EXIT_FAILURE);
    }

    maxExtent_ = surfaceCapabilities.maxImageExtent;
    minExtent_ = surfaceCapabilities.minImageExtent;

    bool fgSwapchainActive = Renderer::options.dlssFrameGenerationActive;

    // Determine number of images for swap chain
    imageCount_ = surfaceCapabilities.minImageCount + (fgSwapchainActive ? 2u : 1u);
    uint32_t preferredImageCap = fgSwapchainActive ? 4u : 3u;
    imageCount_ = std::max(surfaceCapabilities.minImageCount, std::min(imageCount_, preferredImageCap));
    if (surfaceCapabilities.maxImageCount != 0 && imageCount_ > surfaceCapabilities.maxImageCount) {
        imageCount_ = surfaceCapabilities.maxImageCount;
    }

#ifdef DEBUG
    swapchainCout() << "using " << imageCount_ << " images for swap chain" << std::endl;
#endif

    // Find supported surface formats
    uint32_t formatCount;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(), &formatCount,
                                             nullptr) != VK_SUCCESS ||
        formatCount == 0) {
        swapchainCerr() << "failed to get number of supported surface formats" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(), &formatCount,
                                             surfaceFormats.data()) != VK_SUCCESS) {
        swapchainCerr() << "failed to get supported surface formats" << std::endl;
        exit(EXIT_FAILURE);
    }

// Select a surface format
#ifdef DEBUG
    for (int i = 0; i < formatCount; i++) {
        swapchainCout() << "Supported Format: " << surfaceFormats[i].format
                        << " ColorSpace: " << surfaceFormats[i].colorSpace << std::endl;
    }
#endif
    surfaceFormat_ = chooseSurfaceFormat(surfaceFormats);

    // Find supported present modes
    uint32_t presentModeCount;
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &presentModeCount, nullptr) != VK_SUCCESS ||
        presentModeCount == 0) {
        swapchainCerr() << "failed to get number of supported presentation modes" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &presentModeCount, presentModes.data()) != VK_SUCCESS) {
        swapchainCerr() << "failed to get supported presentation modes" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Choose presentation mode (preferring MAILBOX ~= triple buffering)
    presentMode_ = choosePresentMode(presentModes);

    // Select swap chain size
    extent_ = chooseSwapExtent(surfaceCapabilities, window_->width(), window_->height());

    // Determine transformation to use (preferring no transform)
    VkSurfaceTransformFlagBitsKHR surfaceTransform;
    if (surfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        surfaceTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        surfaceTransform = surfaceCapabilities.currentTransform;
    }

    // Finally, create the swap chain
    VkSwapchainKHR oldSwapchain = swapchain_;

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = window_->vkSurface();
    createInfo.minImageCount = imageCount_;
    createInfo.imageFormat = surfaceFormat_.format;
    createInfo.imageColorSpace = surfaceFormat_.colorSpace;
    createInfo.imageExtent = extent_;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT; // TODO: cancel VK_IMAGE_USAGE_SAMPLED_BIT
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = surfaceTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode_;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = oldSwapchain;

    if (vkCreateSwapchainKHR(device_->vkDevice(), &createInfo, nullptr, &swapchain_) != VK_SUCCESS) {
        swapchainCerr() << "failed to create swap chain" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        swapchainCout() << "created swap chain" << std::endl;
#endif
    }

    if (oldSwapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device_->vkDevice(), oldSwapchain, nullptr); }

    // Store the images used by the swap chain
    // Note: these are the images that swap chain image indices refer to
    // Note: actual number of images may differ from requested number, since it's a lower bound
    uint32_t actualImageCount = 0;
    if (vkGetSwapchainImagesKHR(device_->vkDevice(), swapchain_, &actualImageCount, nullptr) != VK_SUCCESS ||
        actualImageCount == 0) {
        swapchainCerr() << "failed to acquire number of swap chain images" << std::endl;
        exit(EXIT_FAILURE);
    }
#ifdef DEBUG
    std::cout << "actualImageCount: " << actualImageCount << std::endl;
#endif
    imageCount_ = actualImageCount;

    std::vector<VkImage> images(actualImageCount);
    if (vkGetSwapchainImagesKHR(device_->vkDevice(), swapchain_, &actualImageCount, images.data()) != VK_SUCCESS) {
        swapchainCerr() << "failed to acquire swap chain images" << std::endl;
        exit(EXIT_FAILURE);
    }
    swapchainImages_.clear();
    for (int i = 0; i < actualImageCount; i++) {
        swapchainImages_.push_back(
            SwapchainImage::create(device_, images[i], extent_.width, extent_.height, surfaceFormat_.format));
    }

#ifdef DEBUG
    swapchainCout() << "acquired swap chain images" << std::endl;
#endif

}

vk::Swapchain::~Swapchain() {
    vkDestroySwapchainKHR(device_->vkDevice(), swapchain_, nullptr);

#ifdef DEBUG
    swapchainCout() << "swapchain deconstructed" << std::endl;
#endif
}

VkSwapchainKHR &vk::Swapchain::vkSwapchain() {
    return swapchain_;
}

VkExtent2D &vk::Swapchain::vkExtent() {
    return extent_;
}

VkExtent2D &vk::Swapchain::vkMaxExtent() {
    return maxExtent_;
}

VkExtent2D &vk::Swapchain::vkMinExtent() {
    return minExtent_;
}

VkSurfaceFormatKHR &vk::Swapchain::vkSurfaceFormat() {
    return surfaceFormat_;
}

std::vector<std::shared_ptr<vk::SwapchainImage>> &vk::Swapchain::swapchainImages() {
    return swapchainImages_;
}

uint32_t vk::Swapchain::imageCount() {
    return imageCount_;
}

bool vk::Swapchain::isHdrOutputActive() const {
    return surfaceFormat_.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
           surfaceFormat_.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
}

