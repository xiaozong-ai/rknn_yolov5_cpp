// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dirent.h>
#include <iostream>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
           "type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
           attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const std::string filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename.c_str(), "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;

    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

// 进行指标计算
static void calculate_mAP()
{
    
}

// 前处理
static void preprocess(rknn_input *inputs, cv::Mat ori_mat, model_info m_info)
{
    int ret;
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_info.in_width * m_info.in_height * m_info.in_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    void *resize_buf = nullptr;

    if (ori_mat.cols != m_info.in_width || ori_mat.rows != m_info.in_height) {
        printf("resize with RGA!\n");
        resize_buf = malloc(m_info.in_width * m_info.in_height * m_info.in_channel);
        memset(resize_buf, 0x00, m_info.in_width * m_info.in_height * m_info.in_channel);

        src = wrapbuffer_virtualaddr((void *)ori_mat.data, ori_mat.cols, ori_mat.rows, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf, m_info.in_width, m_info.in_height, RK_FORMAT_RGB_888);
        ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret)
        {
            printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
            return;
        }
        IM_STATUS STATUS = imresize(src, dst);
        inputs[0].buf = resize_buf;
    } else {
        inputs[0].buf = (void *)ori_mat.data;
    }
}

// 推理及相关处理
static void inference(rknn_context ctx, model_info m_info, rknn_input_output_num io_num, rknn_tensor_attr *output_attrs, rknn_input *inputs, int ori_img_w, int ori_img_h)
{
    int ret;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    struct timeval start_time, stop_time;
    float scale_w;
    float scale_h;
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!m_info.is_quant);
    }

    ret = rknn_run(ctx, NULL);
    if (0 > ret) {
        printf("rknn_run fail! ret=%d\n", ret);
        return;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return;
    }
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    scale_w = (float)m_info.in_width / ori_img_w;
    scale_h = (float)m_info.in_height / ori_img_h;

    for (int i = 0; i < io_num.n_output; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post_process_fp32((float *)outputs[0].buf, (float *)outputs[1].buf, (float *)outputs[2].buf, m_info.in_height, m_info.in_width,
                      box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

}

// 遍历测试集所有图像进行推理
static void traverse_test_dataset(const std::string &testdataset_dir_path, rknn_context ctx, model_info m_info, rknn_input_output_num io_num, rknn_tensor_attr *output_attrs)
{
    DIR *dir;
    struct dirent *entry;
    std::string temp_img_name;
    std::string temp_img_path;
    std::string img_dir = testdataset_dir_path + "/images";
    std::string label_dir = testdataset_dir_path + "/labels";

    cv::Mat origin_img;
    cv::Mat cvt_color_img;

    rknn_input inputs[1];

    dir = opendir(testdataset_dir_path.c_str());
    if (dir == nullptr)
    {
        printf("open test dataset dir failed!!!\n");
        return;
    }

    while (1)
    {
        if ((entry = readdir(dir)) == nullptr) {
            break;
        }

        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        temp_img_name = entry->d_name;
        if (temp_img_name.find(".jpg") != temp_img_name.npos) {
            temp_img_path = testdataset_dir_path + "/images/" + temp_img_name;

            origin_img = cv::imread(temp_img_path, 1);      // 读取图片
            if (!origin_img.data)
            {
                printf("cannot imread %s!!!\n", temp_img_path);
                return;
            }

            cv::cvtColor(origin_img, cvt_color_img, cv::COLOR_BGR2RGB);

            memset(inputs, 0, sizeof(inputs));

            // 前处理
            preprocess(inputs, cvt_color_img, m_info);

            // 推理&后处理
            inference(ctx, m_info, io_num, output_attrs, inputs, cvt_color_img.cols, cvt_color_img.rows);
        }
    }
    
    closedir(dir);
}
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char **argv)
{
    int ret;
    std::string model_name = "model/RK3588/yolov5m6-fp16-768-1280.rknn";    // 模型相对路径
    std::string test_dataset_path = "";
    rknn_context ctx;
    int model_data_size = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    model_info m_info;

    unsigned char *model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 查询算法模型输入和模型输出信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
        m_info.is_quant = true;
    } else {
        m_info.is_quant = false;
    }

    // 组装模型信息
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        m_info.in_channel = input_attrs[0].dims[1];
        m_info.in_height = input_attrs[0].dims[2];
        m_info.in_width = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        m_info.in_height = input_attrs[0].dims[1];
        m_info.in_width = input_attrs[0].dims[2];
        m_info.in_channel = input_attrs[0].dims[3];
    }

    // rknn_context ctx;
    traverse_test_dataset(test_dataset_path, ctx, m_info, io_num, output_attrs);
}