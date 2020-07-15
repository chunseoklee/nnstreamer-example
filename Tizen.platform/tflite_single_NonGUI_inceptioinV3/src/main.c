/**
 * @file main.c
 * @date 19 Feb 2020
 * @brief TIZEN Native Example App of Tflite Inception V3 with single capi
 * @see  https://github.com/nnsuite/nnstreamer
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */

#include <glib.h>
#include <stdint.h>
#include <stdio.h>
#include <nnstreamer.h>
#include <nnstreamer-single.h>

#include <sys/time.h>

long timediff(clock_t t1, clock_t t2) {
    long elapsed;
    elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
    return elapsed;
}

#define MODEL_PATH  1
#define FILE_PATH 2

#define MODEL_WIDTH 299
#define MODEL_HEIGHT  299
#define CH  3
#define LABEL_SIZE  5

gchar * file_path;
gchar * model_path;

/**
 * @brief get the error name rather than code number
 */
char* getErrorName(int status){
  switch(status){
    case TIZEN_ERROR_NONE:
      return "TIZEN_ERROR_NONE";
    case TIZEN_ERROR_INVALID_PARAMETER:
      return "TIZEN_ERROR_INVALID_PARAMETER";
    case TIZEN_ERROR_STREAMS_PIPE:
      return "TIZEN_ERROR_STREAMS_PIPE";
    case TIZEN_ERROR_TRY_AGAIN:
      return "TIZEN_ERROR_TRY_AGAIN";
    case TIZEN_ERROR_UNKNOWN:
      return "TIZEN_ERROR_UNKNOWN";
    case TIZEN_ERROR_TIMED_OUT:
      return "TIZEN_ERROR_TIMED_OUT";
    case TIZEN_ERROR_NOT_SUPPORTED:
      return "TIZEN_ERROR_NOT_SUPPORTED";
    case TIZEN_ERROR_PERMISSION_DENIED:
      return "TIZEN_ERROR_PERMISSION_DENIED";
    case TIZEN_ERROR_OUT_OF_MEMORY:
      return "TIZEN_ERROR_OUT_OF_MEMORY";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief read data source file
 */
 int readInputSource(void **input_buf, size_t data_size)
{
  FILE *fileptr;
  long filelen;
  int ret = -1;

  fileptr = fopen(file_path, "rb");
  fseek(fileptr, 0, SEEK_END);
  filelen = ftell(fileptr);
  rewind(fileptr);

  if(data_size == filelen){
    fread(*input_buf, filelen, 1, fileptr);
    ret = 0;
  }
  fclose(fileptr);

  return ret;
}

/**
 * @brief main function
 */
int main(int argc, char *argv[]){
  ml_single_h single;
  ml_tensors_info_h in_info = NULL;
  ml_tensors_data_h input_data = NULL;
  ml_tensors_data_h output_data = NULL;
  uint16_t *output_buf = NULL;
  void *input_buf = NULL;
  size_t data_size;
  int max = -1;
  int label_num = -1;

  int status;
  file_path = g_strdup_printf("%s", argv[FILE_PATH]);
  model_path = g_strdup_printf("%s", argv[MODEL_PATH]);

  printf("FILE path: %s\n", file_path);
  printf("MODEL path: %s\n", model_path);

  if(!g_file_test (model_path, G_FILE_TEST_EXISTS)){
    printf("[%s] IS NOT EXISTED!!\n", model_path);
    goto finish;
  }

  if(!g_file_test (file_path, G_FILE_TEST_EXISTS)){
    printf("[%s] IS NOT EXISTED!!\n", file_path);
    goto finish;
  }

  input_buf = (uint8_t *)malloc(MODEL_WIDTH * MODEL_HEIGHT * CH * sizeof(uint8_t));
  status = readInputSource(&input_buf, MODEL_WIDTH * MODEL_HEIGHT * CH);
  if(status != ML_ERROR_NONE){
    printf("[%d] FILE READ ERROR\n", __LINE__);
    goto finish;
  }

  status = ml_single_open(&single, model_path, NULL, NULL,
      ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  // status = ml_single_open(&single, model_path, NULL, NULL,
      // ML_NNFW_TYPE_ANY, ML_NNFW_HW_ANY);
   if(status != ML_ERROR_NONE){
    printf("[%d] ERROR: %s\n", __LINE__, getErrorName(status));
    goto finish;
  }

  status = ml_single_get_input_info(single, &in_info);
  if(status != ML_ERROR_NONE){
    printf("[%d] ERROR: %s\n", __LINE__, getErrorName(status));
    goto finish;
  }

  status = ml_tensors_data_create(in_info, &input_data);
  if(status != ML_ERROR_NONE){
    printf("[%d] ERROR: %s\n", __LINE__, getErrorName(status));
    goto finish;
  }

  status = ml_tensors_data_set_tensor_data(input_data, 0, input_buf,
      MODEL_WIDTH * MODEL_HEIGHT * CH);
  if(status != ML_ERROR_NONE){
    printf("[%d] ERROR: %s\n", __LINE__, getErrorName(status));
    goto finish;
  }

  struct timeval start, end;

  for(int i=0;i<5;i++) {
  // start timer.
  gettimeofday(&start, NULL);

  status = ml_single_set_timeout(single, 20000);
  status = ml_single_invoke(single, input_data, &output_data);
  if(status != ML_ERROR_NONE){
    printf("[%d] ERROR: %s\n", __LINE__, getErrorName(status));
    goto finish;
  }

  // stop timer.
  gettimeofday(&end, NULL);
  unsigned long time_in_micros = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
  printf("ml_single_invoke takes : %ld ms\n", time_in_micros );
  }
  status = ml_tensors_data_get_tensor_data(output_data, 0, (void **) &output_buf,
      &data_size);
  if(status != ML_ERROR_NONE){
    printf("[%d] ERROR: %s\n", __LINE__, getErrorName(status));
    goto finish;
  }

  printf("Output Data Size: %llu\n", data_size);
  for (int i = 0; i < LABEL_SIZE; i++) {
    if (output_buf[i] > 0 && output_buf[i] > max) {
      max = output_buf[i];
      label_num = i;
      printf("max: %d, labelnum: %d\n", max, label_num);
    }
  }
  printf(">>> RESULT LABEL NUMBER: %d\n", label_num);

finish:
  ml_tensors_data_destroy(output_data);
  ml_tensors_data_destroy(input_data);
  ml_tensors_info_destroy(in_info);
  ml_single_close(single);
  g_free(model_path);
  g_free(file_path);
  g_free(input_buf);

  return 0;
}
