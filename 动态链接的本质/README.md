# 动态链接的本质

学过计算机的同学都听说过一句话
```
冯诺伊曼架构（Von Neumann Architecture）是现代计算机设计的基础架构之一，该架构的核心思想是将程序指令和数据存储在同一存储器中，使得计算机能够灵活地存储和执行程序。
```

之前对这句话理解不深，直到用到了动态链接才明白。

当我们定义一个数组
```c
uint8_t a[] = {1, 2, 3, 4, 5};
```
时，内存中就有了一段数据，起始地址是a，长度为5字节。

当我们实现了一个函数
```c
int add(int a, int b) {
    return a + b;
}
```
时，得到了什么呢？既然冯诺伊曼架构下一切都是内存中的数据，那么这个函数应该也是一段数据，起始地址是add，至于长度嘛，emmm，不清楚，但我盲猜不会超过128字节？

函数到底是不是一段数据呢？做个实验就知道了：

```cpp
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/mman.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    // normal call
    printf("add(1, 2) = %d\n", add(1, 2));

    // create a large enough memory area
    const size_t size = 128;
    // NOTICE: 3rd argument: allow to read/write data and execute instructions
    void* buffer = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(buffer != MAP_FAILED);
    // copy <add> there
    memcpy(buffer, (void*)add, size);

    // pointer to the copied <add>, then view it as a function (int, int)->int
    int (*cloned_add)(int, int) = (int(*)(int, int))buffer;
    printf("cloned_add(3, 4) = %d\n", cloned_add(3, 4));

    return 0;
}
```

编译执行，不出所料，得到
```
> g++ test.cpp -o test && ./test
add(1, 2) = 3
cloned_add(3, 4) = 7
```

cloned_add()与add()拥有相同的功能，而它是我们运行时动态创建出来的。这说明函数也可以像数组那样用memcpy()读取、写入。或者更确切地说，<b>函数就是数组</b>。

这么说来，如果我拍脑袋想到一段幸运数字，把它当作函数，它是否也能执行呢？答：只要它够幸运，能被CPU理解成有效的指令，那就能执行，不信你看：

```cpp
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/mman.h>

int main() {
    // the magic numbers I dreamed of :-)
    unsigned char code[] = {
        0xB8, 0x2A, 0x00, 0x00, 0x00,
        0xC3                        
    };

    // create a memory area with RWX
    const size_t size = sizeof(code);
    void* buffer = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(buffer != MAP_FAILED);

    // copy <code> there so we can execute it
    memcpy(buffer, (void*)code, size);

    // view it as a function (void)->int
    int (*func)(void) = (int(*)(void))buffer;
    printf("magic function: %d\n", func());

    return 0;
}
```

执行后输出42。哈哈哈，这段数字当然不是我梦见的，它是汇编指令
```
mov eax, 42
ret
```
的直白翻译罢了。

聪明的你一定想到了，如果我写了一个程序，里面包含了很多函数，编译之后，把这些函数当作数组读出来，存到一个文件里。然后另一个程序中读取这些文件，加载到内存中，就可以直接复用这些函数了？可不是嘛，这就是动态链接库（Windows上的dll（Dynamic Link Library），Linux上的so（Shared Object））的原理，早出生100年你也是名垂青史的计算机先驱了啊！

现代的动态链接技术就是在此本质原理的基础上，标准化、规范化了很多细节，比如定义了格式、需要包含的信息（函数名、长度、调用约定（比如多个参数从左往右入栈还是从右往左））等，并且实现了相关的API方便开发者加载、访问动态库及其中的函数。有兴趣可以学习Linux上头文件`dlfcn.h`中定义的API，比如`dlopen`、`dlsym`等等。
