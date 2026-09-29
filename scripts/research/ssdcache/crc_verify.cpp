// crc32 values must not move: the store format and the Python cross check both depend on them
#include <cstdint>
#include <cstdio>
#include <vector>

uint32_t llama_io_crc32(const void * data, size_t size);

int main() {
    std::vector<uint8_t> buf(70000);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t) ((i * 131 + i / 7 + 17) & 0xFF);
    const size_t sizes[] = {0,1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,1000,4096,4084,65536,69999,70000};
    for (size_t s : sizes) printf("%zu %08x\n", s, (unsigned) llama_io_crc32(buf.data(), s));
    return 0;
}
