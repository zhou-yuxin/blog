#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>

#include <random>
#include <algorithm>

int main(int argc, const char** argv) {
    uint32_t n;
    if(!(argc == 4 && sscanf(argv[3], "%u", &n) == 1)) {
        fprintf(stderr, "usage: %s <input> <output> <n>\n", argv[0]);
        return 1;
    }
    const char* fpath_input = argv[1];
    int input = open(fpath_input, O_RDONLY);
    if(input < 0) {
        fprintf(stderr, "failed to open '%s'\n", fpath_input);
        return 1;
    }
    const char* fpath_output = argv[2];
    int output = open(fpath_output, O_WRONLY | O_CREAT, 0644);
    if(output < 0) {
        fprintf(stderr, "failed to open '%s'\n", fpath_output);
        return 1;
    }
    uint32_t n_vector, n_dim;
    if(read(input, &n_vector, 4) != 4) {
        fprintf(stderr, "failed to read number of vectors\n");
        return 1;
    }
    if(read(input, &n_dim, 4) != 4) {
        fprintf(stderr, "failed to read dimension of vectors\n");
        return 1;
    }
    printf("n_vector = %u, n_dim = %u\n", n_vector, n_dim);
    if(write(output, &n, 4) != 4) {
        fprintf(stderr, "failed to write number of vectors\n");
        return 1;
    }
    if(write(output, &n_dim, 4) != 4) {
        fprintf(stderr, "failed to write dimension of vectors\n");
        return 1;
    }
    printf("randomly select %u vectors\n", n);
    uint32_t* ids = new uint32_t[n_vector];
    for(uint32_t i = 0; i < n_vector; i++) {
        ids[i] = i;
    }
    std::random_device random;
    std::mt19937 mt(random());
    std::shuffle(ids, ids + n_vector, mt);
    size_t vector_size = sizeof(float) * n_dim;
    void* buffer = malloc(vector_size);
    for(uint32_t i = 0; i < n; i++) {
        uint32_t j = ids[i];
        if(pread(input, buffer, vector_size, 8 + vector_size * j) != vector_size) {
            fprintf(stderr, "failed to read vector %u\n", j);
            return 1;
        }
        if(write(output, buffer, vector_size) != vector_size) {
            fprintf(stderr, "failed to write vector %u\n", i);
            return 1;
        }
        if(i % 100 == 0) {
            printf("\r%u / %u = %.2f%%                      ", i, n, 100.0f * i / n);
            fflush(stdout);
        }
    }
    printf("\n");
    delete[] ids;
    close(input);
    close(output);
    return 0;
}
