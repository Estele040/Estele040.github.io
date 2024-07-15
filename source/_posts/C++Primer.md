---
title: C++Priemr.md
date: 2023-09-01 21:12:34
tags:
categories:
- C++
---
# C++ Primer

<!--more-->


## 1. 开始

### 输入输出

函数的定义包含四个部分：1. 返回类型；2.函数名；3.一个括号包含的形参类表；4.函数体。

编写程序之后，我们需要编译它。很多PC机上的编译器都具备集成开发环境（IDE）将编译器于其他程序创建的分析工具包装在一起。

在C++中包含了一个全面的标准库来提供IO机制，比如**iostream**库。iostream库包含了两个基础类型的istream和ostream，分别表示输入流和输出流。一个流就是一个字符序列，从IO设备里读取或写入IO设备的。

| 对象 |            |
| ---- | ---------- |
| cin  | 标准输入   |
| cout | 标准输出   |
| cerr | 标准错误   |
| clog | 一般性信息 |
| <<   | 输出运算符 |
| >>   | 输入运算符 |

程序示例：

```c++
#include<iostream>
int main()
{
    std::cout << "Enter two numbers:" << std::endl;
    int v1 = 0, v2 = 0;
    std::cin >> v1 >>v2;
    std::cout << "The sum of" << v1 << "and" << v2
              << "is" << v1 + v2 << std:endl;
    return 0;
}
```

程序中的`std::`指出了名字cout和endl是定义在名为std命名空间中的。

endl（操纵符），结束当前行，并将与设备关联的缓冲区（buffer）中的内容刷到设备中。



### 注释

//：单行注释

/**/：多行注释



### 读取不定的输入数据

```c++
#include<iostream>
int main()
{
    int sum = 0,value = 0;
    //读取数据直到文件尾，计算所有读入的值的和
    while(std::cin >> value)
        sum += value;
    std::cout << "Sum is:" << sum <<std::endl;
    return 0;
}
```



## 2 变量和基本类型

### 2.5处理类型

#### `typedef`

```c++
typedef double wages; //wages是double的同义词
typedef wages base, *p; //base是double的同义词，p是double* 的同义词
```



#### `别名声明`

```c++
using SI = Sales_item; //SI是sales_item的同义词
```



#### `auto`类型说明符

auto类型说明符号的作用是让编译器去分析表达式所属的类型。那么显然，auto定义的变量必须有初始值。

```c++
auto item = val1 + val2;
```

如果val1和val2是类Sales_item的对象，则item的类型就是Sales_item；如果这两个变量的类型是double，则auto的类型就是double，以此类推。

使用auto也能在一条语句上声明多个变量，但条件是所有变量的初始数据类型都一样。

#### auto的使用

首先，我们所熟知的，使用引用其实是使用引用的对象，特别是当引用被用作初始值时，真正参与初始化的其实是引用对象的值。此时编译器以引用对象的类型作为auto的类型：

```c++
int i = 0, &r = i;
auto a = r; //a是一个整数（r是i的别名，而i是一个整数）
```

其次，auto一般会忽略掉顶层const，同时底层const则会保留下来，比如当初始值是一个指向常量的指针时：

```c++
const int ci = i,&cr = ci;
auto b = ci;	//b是一个整数(ci的顶层const特性被忽略掉了)
auto c = cr;	//c是一个整数(cr是ci的别名，ci本身是一个顶层const)
auto d = &i;	//d是一个指向整形的指针(整数的地址就是指向整数的指针)
auto r = &ci;	//e是一个指向整数常量的指针(对常量对象取缔值是一种底层const)
```

如果希望推断出的auto类型是一个顶层const，需要说明指出：

```c++
const auto f = ci;	//ci的推演类型是int，f是const int
```

还可以将引用类型设为auto，此时原来的初始化规则仍然适用：

```c++
auto &g = ci;	//g是一个整形常量引用，绑定到ci
auto &h = 42;	//错误：不能为非常量引用绑定字面值
const auto &j = 42; //正确：可以为常量引用绑定字面值
```






