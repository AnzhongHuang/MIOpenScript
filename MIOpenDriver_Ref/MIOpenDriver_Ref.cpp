#include <torch/extension.h>

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#define LOG_ENABLE 0

namespace py = pybind11;

torch::Tensor gpu_convolution_reference(
    torch::Tensor output,
    torch::Tensor weight,
    torch::Tensor input,
    int64_t padding,
    int64_t stride,
    int64_t dilation,
    int64_t group,
    int64_t solution_id,
    int64_t operation,
    std::string type)
{
    // Validate inputs
    if (output.dim() != 4 || weight.dim() != 4 || input.dim() != 4)
    {
        throw std::invalid_argument("Expected 4D tensors for output / weight / input");
    }

    // Check if tensors are on GPU
    if (!output.is_cuda() || !weight.is_cuda())
    {
        throw std::runtime_error("Tensors must be on CUDA device for MIOpen");
    }

    // Initialize MIOpen handle
    miopenHandle_t handle;
    miopenStatus_t status = miopenCreate(&handle);
    if (status != miopenStatusSuccess)
    {
        throw std::runtime_error("Failed to create MIOpen handle");
    }

    try
    {
        // Create tensor descriptors
        miopenTensorDescriptor_t output_desc, weight_desc, input_desc;
        status = miopenCreateTensorDescriptor(&output_desc);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to create output descriptor");
        }
        status = miopenCreateTensorDescriptor(&weight_desc);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to create weight descriptor");
        }
        status = miopenCreateTensorDescriptor(&input_desc);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to create input descriptor");
        }

        // Ensure tensors are contiguous and float type
        auto output_contiguous = output.contiguous();
        auto weight_contiguous = weight.contiguous();
        auto input_contiguous = input.contiguous();

        miopenDataType_t data_type = miopenFloat;
        if (type == "fp16")
        {
            data_type = miopenHalf;
        }
        else if (type == "bfp16")
        {
            data_type = miopenBFloat16;
        }
        else if (type == "int8")
        {
            data_type = miopenInt8;
        }

        // Set tensor descriptors
        status = miopenSet4dTensorDescriptor(output_desc, data_type,
                                             output_contiguous.size(0), output_contiguous.size(1),
                                             output_contiguous.size(2), output_contiguous.size(3));
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set output descriptor");
        }

        status = miopenSet4dTensorDescriptor(weight_desc, data_type,
                                             weight_contiguous.size(0), weight_contiguous.size(1),
                                             weight_contiguous.size(2), weight_contiguous.size(3));
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set weight descriptor");
        }

        status = miopenSet4dTensorDescriptor(input_desc, data_type,
                                             input_contiguous.size(0), input_contiguous.size(1),
                                             input_contiguous.size(2), input_contiguous.size(3));
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set input descriptor");
        }

        // Create convolution descriptor
        miopenConvolutionDescriptor_t conv_desc;
        status = miopenCreateConvolutionDescriptor(&conv_desc);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to create convolution descriptor");
        }

        status = miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                                 padding, padding, stride, stride,
                                                 dilation, dilation);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to initialize convolution descriptor");
        }

        status = miopenSetConvolutionGroupCount(conv_desc, group);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set convolution groupCount");
        }

        // Create output tensor with proper device and type
        // auto grad_input = torch::empty(input_shape,
        //                                torch::TensorOptions()
        //                                    .dtype(torch::kFloat32)
        //                                    .device(grad_output.device()));

        // Print tensor info for debugging
#if LOG_ENABLE
        printf("output shape: [%ld, %ld, %ld, %ld]\n",
               output_contiguous.size(0), output_contiguous.size(1),
               output_contiguous.size(2), output_contiguous.size(3));
        printf("weight shape: [%ld, %ld, %ld, %ld]\n",
               weight_contiguous.size(0), weight_contiguous.size(1),
               weight_contiguous.size(2), weight_contiguous.size(3));
        printf("input shape: [%ld, %ld, %ld, %ld]\n",
               input_contiguous.size(0), input_contiguous.size(1),
               input_contiguous.size(2), input_contiguous.size(3));
#endif

        // Get workspace size first
        size_t workspace_size = 0;
        status = miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle, output_desc, weight_desc, conv_desc, input_desc, &workspace_size);

        void *workspace = nullptr;
        if (workspace_size > 0)
        {
            hipMalloc(&workspace, workspace_size);
        }

        // Call MIOpen backward data convolution with proper error checking
        if (operation == 1)
        {

            if (data_type == miopenFloat)
            {
                status = miopenConvolutionForwardImmediate(
                    handle,
                    weight_desc, weight_contiguous.data_ptr<float>(),
                    input_desc, input_contiguous.data_ptr<float>(),
                    conv_desc,
                    output_desc, output_contiguous.data_ptr<float>(),
                    workspace, workspace_size, 85);
            }
            else if (data_type == miopenHalf)
            {
                status = miopenConvolutionForwardImmediate(
                    handle,
                    weight_desc, weight_contiguous.data_ptr<c10::Half>(),
                    input_desc, input_contiguous.data_ptr<c10::Half>(),
                    conv_desc,
                    output_desc, output_contiguous.data_ptr<c10::Half>(),
                    workspace, workspace_size, 85);
            }
            else if (data_type == miopenBFloat16)
            {
                status = miopenConvolutionForwardImmediate(
                    handle,
                    weight_desc, weight_contiguous.data_ptr<c10::BFloat16>(),
                    input_desc, input_contiguous.data_ptr<c10::BFloat16>(),
                    conv_desc,
                    output_desc, output_contiguous.data_ptr<c10::BFloat16>(),
                    workspace, workspace_size, 85);
            }
            else if (data_type == miopenInt8)
            {
                status = miopenConvolutionForwardImmediate(
                    handle,
                    weight_desc, weight_contiguous.data_ptr<int8_t>(),
                    input_desc, input_contiguous.data_ptr<int8_t>(),
                    conv_desc,
                    output_desc, output_contiguous.data_ptr<int8_t>(),
                    workspace, workspace_size, 85);
            }

            if (status != miopenStatusSuccess)
            {
                throw std::runtime_error("MIOpen forward convolution failed with status: " + std::to_string(status));
            }
        }
        else if (operation == 2)
        {
            if (data_type == miopenFloat)
            {
                status = miopenConvolutionBackwardDataImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<float>(),
                    weight_desc, weight_contiguous.data_ptr<float>(),
                    conv_desc,
                    input_desc, input_contiguous.data_ptr<float>(),
                    workspace, workspace_size, 86);
            }
            else if (data_type == miopenHalf)
            {
                status = miopenConvolutionBackwardDataImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<c10::Half>(),
                    weight_desc, weight_contiguous.data_ptr<c10::Half>(),
                    conv_desc,
                    input_desc, input_contiguous.data_ptr<c10::Half>(),
                    workspace, workspace_size, 86);
            }
            else if (data_type == miopenBFloat16)
            {
                status = miopenConvolutionBackwardDataImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<c10::BFloat16>(),
                    weight_desc, weight_contiguous.data_ptr<c10::BFloat16>(),
                    conv_desc,
                    input_desc, input_contiguous.data_ptr<c10::BFloat16>(),
                    workspace, workspace_size, 86);
            }
            else if (data_type == miopenInt8)
            {
                status = miopenConvolutionBackwardDataImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<int8_t>(),
                    weight_desc, weight_contiguous.data_ptr<int8_t>(),
                    conv_desc,
                    input_desc, input_contiguous.data_ptr<int8_t>(),
                    workspace, workspace_size, 86);
            }

            if (status != miopenStatusSuccess)
            {
                throw std::runtime_error("MIOpen backward data convolution failed with status: " + std::to_string(status));
            }
        }
        else if (operation == 4)
        {
            if (data_type == miopenFloat)
            {
                status = miopenConvolutionBackwardWeightsImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<float>(),
                    input_desc, input_contiguous.data_ptr<float>(),
                    conv_desc,
                    weight_desc, weight_contiguous.data_ptr<float>(),
                    workspace, workspace_size, 87);
            }
            else if (data_type == miopenHalf)
            {
                status = miopenConvolutionBackwardWeightsImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<c10::Half>(),
                    input_desc, input_contiguous.data_ptr<c10::Half>(),
                    conv_desc,
                    weight_desc, weight_contiguous.data_ptr<c10::Half>(),
                    workspace, workspace_size, 87);
            }
            else if (data_type == miopenBFloat16)
            {
                status = miopenConvolutionBackwardWeightsImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<c10::BFloat16>(),
                    input_desc, input_contiguous.data_ptr<c10::BFloat16>(),
                    conv_desc,
                    weight_desc, weight_contiguous.data_ptr<c10::BFloat16>(),
                    workspace, workspace_size, 87);
            }
            else if (data_type == miopenInt8)
            {
                status = miopenConvolutionBackwardWeightsImmediate(
                    handle,
                    output_desc, output_contiguous.data_ptr<int8_t>(),
                    input_desc, input_contiguous.data_ptr<int8_t>(),
                    conv_desc,
                    weight_desc, weight_contiguous.data_ptr<int8_t>(),
                    workspace, workspace_size, 87);
            }

            if (status != miopenStatusSuccess)
            {
                throw std::runtime_error("MIOpen backward weights convolution failed with status: " + std::to_string(status));
            }
        }
        else
        {
            throw std::runtime_error("The Operation is set error[Forward-0 / BackwardData-2 / BackwardWeight-4]");
        }

        // Clean up workspace
        if (workspace)
        {
            hipFree(workspace);
        }

        // Cleanup descriptors
        miopenDestroyConvolutionDescriptor(conv_desc);

        miopenDestroyTensorDescriptor(output_desc);
        miopenDestroyTensorDescriptor(weight_desc);
        miopenDestroyTensorDescriptor(input_desc);

        miopenDestroy(handle);

        if (operation == 1)
        {
            return output_contiguous;
        }
        else if (operation == 2)
        {
            return input_contiguous;
        }
        else if (operation == 4)
        {
            return weight_contiguous;
        }

        return torch::Tensor{};
    }
    catch (...)
    {
        // Cleanup handle on exception
        miopenDestroy(handle);
        throw;
    }
}

// Additional utility function for error checking
std::string get_miopen_version()
{
    size_t major, minor, patch;
    miopenGetVersion(&major, &minor, &patch);
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

// Pybind11 module definition
PYBIND11_MODULE(MIOpenDriver_Ref, m)
{
    m.doc() = "MIOpen convolution backward data operations";

    m.def("gpu_convolution_reference", &gpu_convolution_reference,
          "Perform convolution reference operation using MIOpen",
          py::arg("grad_output"),
          py::arg("weight"),
          py::arg("input"),
          py::arg("padding"),
          py::arg("stride"),
          py::arg("dilation"),
          py::arg("group"),
          py::arg("solution_id"),
          py::arg("operation"),
          py::arg("type"));

    m.def("get_miopen_version", &get_miopen_version,
          "Get MIOpen library version");
}