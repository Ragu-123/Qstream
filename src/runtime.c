#include "qstream.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int qs_model_open(const char *path, qs_model_t *model) {
  if (!path || !model) {
    return -1;
  }
  memset(model, 0, sizeof(*model));
  model->fd = open(path, O_RDONLY);
  if (model->fd < 0) {
    return -2;
  }
  if (qsf_read_header(path, &model->header) != 0) {
    close(model->fd);
    model->fd = -1;
    return -3;
  }
  model->index_entries = (qsf_layer_index_entry_t *)calloc(
      model->header.num_layers, sizeof(qsf_layer_index_entry_t));
  if (!model->index_entries) {
    qs_model_close(model);
    return -4;
  }
  if (qsf_read_layer_index(path, &model->header, model->index_entries,
                           model->header.num_layers) != 0) {
    qs_model_close(model);
    return -5;
  }
  return 0;
}

void qs_model_close(qs_model_t *model) {
  if (!model) {
    return;
  }
  if (model->fd >= 0) {
    close(model->fd);
  }
  free(model->index_entries);
  memset(model, 0, sizeof(*model));
  model->fd = -1;
}

int qs_stream_init(const qs_model_t *model, qs_layer_stream_t *stream) {
  if (!model || !stream || model->header.num_layers == 0) {
    return -1;
  }
  memset(stream, 0, sizeof(*stream));
  stream->model = model;
  uint32_t max_comp = 0;
  for (uint32_t i = 0; i < model->header.num_layers; ++i) {
    if (model->index_entries[i].compressed_size > max_comp) {
      max_comp = model->index_entries[i].compressed_size;
    }
  }
  if (max_comp == 0) {
    max_comp = 1;
  }
  for (int b = 0; b < 2; ++b) {
    stream->buffers[b] = (uint8_t *)malloc(max_comp);
    if (!stream->buffers[b]) {
      qs_stream_destroy(stream);
      return -2;
    }
    stream->capacities[b] = max_comp;
  }
  stream->active = -1;
  stream->loaded_layer = UINT32_MAX;
  return 0;
}

void qs_stream_destroy(qs_layer_stream_t *stream) {
  if (!stream) {
    return;
  }
  free(stream->buffers[0]);
  free(stream->buffers[1]);
  memset(stream, 0, sizeof(*stream));
  stream->active = -1;
}

int qs_stream_load_layer(qs_layer_stream_t *stream, uint32_t layer) {
  if (!stream || !stream->model || layer >= stream->model->header.num_layers) {
    return -1;
  }
  int target = (stream->active == 0) ? 1 : 0;
  qsf_layer_index_entry_t e = stream->model->index_entries[layer];
  if (e.compressed_size > stream->capacities[target]) {
    return -2;
  }
  ssize_t read_n = pread(stream->model->fd, stream->buffers[target],
                         e.compressed_size, (off_t)e.layer_offset);
  if (read_n < 0 || (uint32_t)read_n != e.compressed_size) {
    return -3;
  }
  stream->active = target;
  stream->loaded_layer = layer;
  stream->loaded_size = (size_t)e.compressed_size;
  return 0;
}
