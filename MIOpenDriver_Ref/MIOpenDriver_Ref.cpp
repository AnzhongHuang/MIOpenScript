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

namespace py = pybind11;

torch::Tensor gpu_convolution_reference(
    torch::Tensor grad_output,
    torch::Tensor weight,
    std::vector<int64_t> input_shape,
    int64_t padding,
    int64_t stride,
    int64_t dilation,
    int64_t group,
    int64_t solution_id,
    int64_t operation)
{

    printf("\nLog_0: Starting gpu_convolution_reference\n");

    // Validate inputs
    if (grad_output.dim() != 4 || weight.dim() != 4)
    {
        throw std::invalid_argument("Expected 4D tensors for grad_output and weight");
    }
    printf("\nLog_1: Input validation passed\n");

    if (input_shape.size() != 4)
    {
        throw std::invalid_argument("Expected input_shape to have 4 dimensions");
    }
    printf("\nLog_2: Shape validation passed\n");

    // Check if tensors are on GPU
    if (!grad_output.is_cuda() || !weight.is_cuda())
    {
        throw std::runtime_error("Tensors must be on CUDA device for MIOpen");
    }
    printf("\nLog_2.1: CUDA check passed\n");

    // Get HIP stream from PyTorch's CUDA stream
    // hipStream_t hip_stream = at::hip::getCurrentHIPStream();
    // printf("\nLog_2.2: Got HIP stream\n");

    // Initialize MIOpen handle
    miopenHandle_t handle;
    miopenStatus_t status = miopenCreate(&handle);
    if (status != miopenStatusSuccess)
    {
        printf("miopenCreate failed with status: %d\n", status);
        throw std::runtime_error("Failed to create MIOpen handle");
    }
    printf("\nLog_3: MIOpen handle created\n");

    // // Set stream for MIOpen handle
    // status = miopenSetStream(handle, hip_stream);
    // if (status != miopenStatusSuccess) {
    //     miopenDestroy(handle);
    //     printf("miopenSetStream failed with status: %d\n", status);
    //     throw std::runtime_error("Failed to set MIOpen stream");
    // }
    // printf("\nLog_3.1: MIOpen stream set\n");

    try
    {
        // Create tensor descriptors
        miopenTensorDescriptor_t grad_output_desc, weight_desc, grad_input_desc;
        status = miopenCreateTensorDescriptor(&grad_output_desc);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to create grad_output descriptor");
        }
        status = miopenCreateTensorDescriptor(&weight_desc);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to create weight descriptor");
        }
        status = miopenCreateTensorDescriptor(&grad_input_desc);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to create grad_input descriptor");
        }
        printf("\nLog_4: Tensor descriptors created\n");

        // Ensure tensors are contiguous and float type
        auto grad_output_contiguous = grad_output.contiguous().to(torch::kFloat32);
        auto weight_contiguous = weight.contiguous().to(torch::kFloat32);
        printf("\nLog_4.1: Tensors made contiguous\n");

        // Set tensor descriptors
        status = miopenSet4dTensorDescriptor(grad_output_desc, miopenFloat,
                                             grad_output_contiguous.size(0), grad_output_contiguous.size(1),
                                             grad_output_contiguous.size(2), grad_output_contiguous.size(3));
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set grad_output descriptor");
        }
        printf("\nLog_5: grad_output descriptor set\n");

        status = miopenSet4dTensorDescriptor(weight_desc, miopenFloat,
                                             weight_contiguous.size(0), weight_contiguous.size(1),
                                             weight_contiguous.size(2), weight_contiguous.size(3));
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set weight descriptor");
        }
        printf("\nLog_6: weight descriptor set\n");

        status = miopenSet4dTensorDescriptor(grad_input_desc, miopenFloat,
                                             input_shape[0], input_shape[1],
                                             input_shape[2], input_shape[3]);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set grad_input descriptor");
        }
        printf("\nLog_7: grad_input descriptor set\n");

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
        printf("\nLog_8: Convolution descriptor created and initialized\n");

        status = miopenSetConvolutionGroupCount(conv_desc, group);
        printf("\n group: %d\n", group);
        if (status != miopenStatusSuccess)
        {
            throw std::runtime_error("Failed to set convolution groupCount");
        }

        // Create output tensor with proper device and type
        auto grad_input = torch::empty(input_shape,
                                       torch::TensorOptions()
                                           .dtype(torch::kFloat32)
                                           .device(grad_output.device()));
        auto grad_input_contiguous = grad_input.contiguous();
        printf("\nLog_9: Output tensor created\n");

        // Print tensor info for debugging
        printf("grad_output shape: [%ld, %ld, %ld, %ld]\n",
               grad_output_contiguous.size(0), grad_output_contiguous.size(1),
               grad_output_contiguous.size(2), grad_output_contiguous.size(3));
        printf("weight shape: [%ld, %ld, %ld, %ld]\n",
               weight_contiguous.size(0), weight_contiguous.size(1),
               weight_contiguous.size(2), weight_contiguous.size(3));
        printf("grad_input shape: [%ld, %ld, %ld, %ld]\n",
               input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Get workspace size first
        size_t workspace_size = 0;
        status = miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle, grad_output_desc, weight_desc, conv_desc, grad_input_desc, &workspace_size);

        void *workspace = nullptr;
        if (workspace_size > 0)
        {
            hipMalloc(&workspace, workspace_size);
            printf("Allocated workspace: %zu bytes\n", workspace_size);
        }

        printf("\nLog_9.5: About to call miopenConvolutionBackwardDataImmediate\n");

        // Call MIOpen backward data convolution with proper error checking

        if (operation == 1)
        {
            status = miopenConvolutionForwardImmediate(
                handle,
                weight_desc, weight_contiguous.data_ptr<float>(),
                grad_input_desc, grad_input_contiguous.data_ptr<float>(),
                conv_desc,
                grad_output_desc, grad_output_contiguous.data_ptr<float>(),
                workspace, workspace_size, 85);

            printf("\nLog_10: miopenConvolutionForwardImmediate returned with status: %d\n", status);
            if (status != miopenStatusSuccess)
            {
                printf("MIOpen error status: %d\n", status);
                throw std::runtime_error("MIOpen forward convolution failed with status: " + std::to_string(status));
            }
        }
        else if (operation == 2)
        {
            status = miopenConvolutionBackwardDataImmediate(
                handle,
                grad_output_desc, grad_output_contiguous.data_ptr<float>(),
                weight_desc, weight_contiguous.data_ptr<float>(),
                conv_desc,
                grad_input_desc, grad_input_contiguous.data_ptr<float>(),
                workspace, workspace_size, 86);

            printf("\nLog_10: miopenConvolutionBackwardDataImmediate returned with status: %d\n", status);
            if (status != miopenStatusSuccess)
            {
                printf("MIOpen error status: %d\n", status);
                throw std::runtime_error("MIOpen backward data convolution failed with status: " + std::to_string(status));
            }
        }
        else if (operation == 4)
        {
            status = miopenConvolutionBackwardWeightsImmediate(
                handle,
                grad_output_desc, grad_output_contiguous.data_ptr<float>(),
                grad_input_desc, grad_input_contiguous.data_ptr<float>(),
                conv_desc,
                weight_desc, weight_contiguous.data_ptr<float>(),
                workspace, workspace_size, 87);

            printf("\nLog_10: miopenConvolutionBackwardWeightsImmediate returned with status: %d\n", status);
            if (status != miopenStatusSuccess)
            {
                printf("MIOpen error status: %d\n", status);
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

        miopenDestroyTensorDescriptor(grad_output_desc);
        miopenDestroyTensorDescriptor(weight_desc);
        miopenDestroyTensorDescriptor(grad_input_desc);
        printf("\nLog_11: Descriptors cleaned up\n");

        miopenDestroy(handle);
        printf("\nLog_12: Handle destroyed\n");

        if (operation == 1)
        {
            return grad_output_contiguous;
        }
        else if (operation == 2)
        {
            return grad_input_contiguous;
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
          py::arg("input_shape"),
          py::arg("padding"),
          py::arg("stride"),
          py::arg("dilation"),
          py::arg("group"),
          py::arg("solution_id"),
          py::arg("operation"));

    m.def("get_miopen_version", &get_miopen_version,
          "Get MIOpen library version");
}