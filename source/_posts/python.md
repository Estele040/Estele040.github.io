---
title: python
date: 2023-08-06 14:12:21
categories: #分类
- python
tags:
- python

---
# Python

Python的学习记录。

<!--more-->



## Python基础

### 字符编码

计算机只能处理数字，如果要处理文本，就必须把文本转化成数字才能处理。最早的计算机在设计时采用8个比特(bit)作为一个字节(byte)，所以，一个字节能表示的最大的整数就是255（二进制11111111=十进制255）。如果要表示更大的整数就要采用更多的字节。

为了不与ASCII编码冲突，中国制定了`GB2312`编码，用来把中文编码进去。日本把日文编到`shift_JIS`中等等。各国有各国的标准，就会发生冲突，结果是，在很多语言混合的文本中，显示出来会有乱码。

因此，Unicode字符集应运而生。Unicode把所有语言都统一到一套编码里，这样就不会出现乱码问题了。

ASCII编码是一个字节，而Unicode编码是两个字节。

字母`A`用ASCII编码是十进制的`65`，二进制的`01000001`；

字符`0`用ASCII编码是十进制的`48`，二进制的`00110000`，注意字符`'0'`和整数`0`是不同的；

汉字`中`已经超出了ASCII编码的范围，用Unicode编码是十进制的`20013`，二进制的`01001110 00101101`。

对于单个字符的编码，Python提供了`ord()`函数获取字符的整数表示，`chr()`函数把编码转化为对应字符。

如果知道字符的整数编码，还可以用十六进制这么写str：

```python
>>>'\u4e2d\u6587'
'中文'
```

这两种写法完全等价。

由于Python的字符串是`str`，所以在内存中以Unicode表示，一个字符对应若干个字节。如果要在网络上传输，或者要保存到磁盘上，就需要把`str`变为以字节为单位的`bytes`。

Python对`bytes`类型的数据用带`b`前缀的单引号或双引号表示：

```python
x = b'ABC'
```

纯英文的`str`可以用`ASCII`编码为`bytes`，内容是一样的，含有中文的`str`可以用`UTF-8`编码为`bytes`。含有中文的`str`无法用`ASCII`编码，Python会报错。

在`bytes`中，无法显示为ASCII字符的字节，用`\x##`显示。

反过来，如果我们从网络或磁盘上读取了字节流，那么读到的数据就是`bytes`要把`bytes`变为`str`，就需要用`decode()`方法：

```python
>>>b'ABC'.decode('ascii')
'ABC'
>>>b'\xe4\xb8\xad\xe6\x96\x87',decode('utf-8')
'中文'
```

如果`bytes`中包含无法解码的字节，`decode()`方法会报错。

如果`bytes`中只有一小部分无效的字节，可以传入`errors='ignore'`忽略错误的字节：

由于Python源码是一个文本文件，所以，当你的源代码中包含了中文的时候，在保存源代码时，就需要务必指定保存UTF-8编码。当Python解释器读取源码时，为了让它按`UTF-8`编码读取，我们通常在文件开头写上这两行：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```

第一行注释是为了告诉Linux/OS X 系统，这是一个Python可执行程序，Windows系统会忽略这个注释；

第二行注释是为了告诉Python解释器，按照UTF-8编码读取源代码，否则，你在源代码中写入的中文输入可能会出现乱码。

在Python中，采用的格式化字符串的方式是与C语言一致的，用`%`实现。

| 占位符 | 替换内容     |
| :----- | :----------- |
| %d     | 整数         |
| %f     | 浮点数       |
| %s     | 字符串       |
| %x     | 十六进制整数 |

其中，格式化整数和浮点数还可以制定是否补0和整数与小数的为数：

```python
print('#2d-%02d' % (3,1))
print('%.2f' % 3.1415926)
```

如果你不太确定应该用什么，`%s`永远起作用，它会把任何数据类型转换为字符串：

有些时候，字符串里面的`%`是一个普通字符怎么办？这个时候就需要转义，用`%%`来表示一个`%`：

#### format()

另一种格式化字符串的方法是使用字符串的`format()`方法，它会用传入的参数依次替换占为符`{0}`、`{1}`、……，不过这种方式写起来比%要麻烦的多：

```python
>>> 'Hello, {0}, 成绩提升了 {1:.1f}%'.format('小明', 17.125)
'Hello, 小明, 成绩提升了 17.1%'
```



#### `f-string`

最后一种格式化字符串的方法是使用以`f`开头的字符串，称之为`f-string`，它和普通字符串不同之处在于，字符串如果包含`{xxx}`，就会以对应变量替换：

```python
>>> r = 2.5
>>> s = 3.4 * r ** 2
>>> print(f'The area of a child with radius {r} is {s:.2f}')
```

在上述代码中，`{r}`被变量`r`替换，`{s:.2f}`被变量`{s}`的值替换，并且`:`后面的`.2f`指定了格式化参数，因此，`{s:.2f}`的替换结果是`19.62`。



## 函数

### 定义函数

在Python中，定义一个函数要使用`def`语句，依次写出函数名，括号，括号中的参数和冒号`:`，然后，在缩进块中编写函数体，函数的返回值用`return`语句返回。

我们自定义一个求绝对值的`my_abs`函数为例：

```python
def my_abs(x):
    if x >= 0:
        return x
    else:
        return -x
```

请注意，函数体内部的语句在执行时，一旦执行到`return`时，函数就执行完毕，并将结果返回。因此，函数内部通过条件判断和循环可以实现非常复杂的逻辑。

如果没有`return`语句，函数执行完毕后也会返回结果，只是结果为`None`。`return None`可以简写成`return`。

在Python交互环境中定义函数时，注意Python会出现`...`的提示。函数定义结束后需要按两次回车重新回到`>>>`提示符下。

如果你已经把`my_abs()`的函数定义保存为`abstest.py`文件了，那么，可以在该文件的当前目录下启动Python解释器，用`from abstest import my_abs`来导入`my_abs()`函数，注意`abstest`是文件名（不含`.py`扩展名）：

```python
>>> from abstest import my_abs
>>> my_abs(-9)
9
>>>_
```

#### 空函数

如果想定义一个什么事也不做的空函数，可以用`pass`语句：

```python
def nop():
    pass
```

一个空函数，缺少了`pass`，代码运行就会有语法错误。

#### 参数检查

调用函数时，如果参数个数不对，Python解释器会自动检查出来，并抛出`TypeError`。

但是如果参数类型不对，Python解释器就无法帮我们检查。试试`my_abs`和内置函数``abs`的差别：

```python
>>> my_abs('A')
Traceback (most recent call last):
    File "<stdin>",line 1, in <module>
    File "<stdin>",line 2, in my_abs
TypeError:unorderable types: str() >= int()
>>>abs('A')
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: bad operand type for abs(): 'str'
```

当传入了不恰当的参数时，内置函数`abs`会检查出错误，而我们定义的`my_abs`没有参数检查，会导致`if`语句出错，出错信息和`abs`不一样。所以这个函数定义不够完善。

让我们来修改一下`my_abs`的定义，对参数类型作检查，只允许整数和浮点数类型的参数。数据类型检查可以用内置函数`isinstance()`实现：

```python
def my_abs(x):
    if not isinstance(x,(int,float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x
```





## 函数式编程

### 高阶函数

#### map/reduce

什么是高阶函数？我们以是实际代码为例子，一步步升入概念。

**变量可以指向函数**

以python内置的求绝对值函数`abs()`为例，调用该函数用以下代码：

```python
>>>abs(-10)
10
```

但是如果只写`abs`呢？

```python
>>>abs
<built-in function abs>
```

可见`abs(function)`是函数调用，而abs是函数本身。

要获得函数调用的结果我们可以把函数赋值给变量：

```python
>>>x = abs(-10)
>>>x
10
```

结论：函数本身因为可以赋值给变量，即：变量可以指向函数。

如果一个变量指向了一个函数，那么可否通过改变量来调用这个函数？用代码验证以下：

```python
>>>f = abs
>>>f(-10)
10
```

说明变量`f`现在已经指向了`abs`函数本身。直接调用`abs()`函数和调用`f`完全相同。

**函数名也是变量**

那么函数名是什么呢？函数名其实就是指向函数的变量！对于`abs`这个函数完全可以把`abs`看成变量，它指向一个可以计算绝对值的函数！如果把`abs`指向其他对象会有什么情况发生？

```python
>>> abs = 10
>>> abs(-10)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'int' object is not callable
```

把`abs`指向`10`后，就无法通过`abs(-10)`调用该函数了！因为`abs`这个变量已经不指向求绝对值函数而是指向一个整数`10`！

当然实际代码绝对不能这么写，这里是为了说明函数名也是变量。要恢复`abs`函数，请重启Python交互环境。

注：由于`abs`函数实际上是定义在`import builtins`模块中的，所以要让修改`abs`变量的指向在其它模块也生效，要用`import builtins; builtins.abs = 10`。



### 偏函数

python的`functools`提供了偏函数功能（Partial function）。

假设要转换大量的二进制字符串，每次都传入`int(x,base = 2)`非常麻烦，于是，我们想到可以定义一个函数`int2()`，默认把`base = 2`传进去：

```python
def int2(x,base = 2):
	return int(x,base)

#这样我们进行二进制转换就可以了
int2(1000000)

#Partial function
int2 = functools.partial(int,base = 2)
int2('123456')
64
```

Partial function作用就是，帮助我们把一个函数的某些参数固定住（也就是设置默认值），返回一个新的函数。

创建偏函数时可以接受函数对象、*args、**kw这3个参数。

```python
int2 = functools.partial(int,base = 2)
```

实际上固定了`int()`函数的关键字`base`，也就是：

```python
int2('10010')
```

相当于：

```python
kw = {'base' : 2}
int('10010' , **kw)
```

当传入：

```python
max2 = functools.partial(max,10)
```

实际上会把10作为*args的一部分自动加到左边：

```python
max2(5, 6, 7)
```

相当于：

```python
args = (10, 5, 6, 7)
max(*args)
```

结果为10。

------------------





## 模块

为了编写可维护代码，我们把很多函数分组，分别放到不同的文件里，这样，每个文件包含的代码就相对较少。在python中一个.py文件就可以被称为一个模块（Module）。

使用模块有什么好处？

最大的好处是提高了代码的可维护性。其次，编写代码不必从0开始。

其次，使用模块还可以避免函数名和变量名冲突。相同名字的函数名和变量名可以存在不同模块中。

如果模块名冲突怎么办？Python又引入了按目录来组织模块的方法，称为包（Package）。

每个包目录下都有一个`__init__.py`文件，这个文件是必须存在的，否则，python就会把这个文件当成普通目录而不是包。`__init__.py`可以是空文件，也可以有Python代码，因为`__init__.py`本身就是一个模块，而它的模块名就是`mycompany`。



### 使用模块

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Michael Liao'

import sys

def test():
    args = sys.argv
    if len(args)==1:
        print('Hello, world!')
    elif len(args)==2:
        print('Hello, %s!' % args[1])
    else:
        print('Too many arguments!')

if __name__=='__main__':
    test()
'''
当我们在命令行运行hello模块文件时，Python解释器把一个特殊变量__name__置为__main__，而如果在其他地方导入该hello模块时，if判断将失败，因此，这种if测试可以让一个模块通过命令行运行时执行一些额外的代码，最常见的就是运行测试。
'''

```

代码的第一，二行是标准注释，第一行注释可以让这个`hellow.py`文件直接在Unix/Linux/Mac上运行，第二行注释表示了.py文件本身使用了UTF-8编码；

第四行是一个字符串，表示模块的文档注释，任何模块的第一行字符串都被视为模块文档注释；

第六行`__author__`变量把作者写进去。

后面开始就是真正的代码部分。

使用`sys`模块的第一步就是导入该模块：

```python
import sys
```

导入了sys模块后，我们就有了变量sys指向该模块，利用`sys`这个变量，就可以访问`sys`模块所有的功能。

sys有一个argv变量，用list存储了命令行的所有参数。argv至少有一个元素，因为第一个参数永远是`.py`文件的名称，例如：

运行``python3 hello.py`获得的`sys.argv`就是`['hellow.py']`

运行`python3 hello.py Michael`获得的`sys.argv`就是`['hello.py', 'Michael']`。



如果启动Python交互环境，再导入`hello`模块：

```python
$ python3
>>>import hello
>>>
```

导入时，没有打印`hello world!`因为没有执行`test()`函数。

调用`hello.test()`时，才能打印出`hello world!`:

```python
>>>hello.test()
Hello,world!
```

#### 作用域

在一个模块中，我们可能会定义很多函数和变量，但有的函数我们希望给别人使用，有的仅仅在模块内部使用。在Python中我们是通过`_`前缀来实现的。

正常函数和变量名是公开的（public），可以被直接引用。

类似`__xx__`是特殊变量，可以被直接引用，但有特殊用途，比如上面的`__author__`，`__name__`就是特殊变量，`hello`模块定义的文档注释也可以用特殊变量`__doc__`访问，我们自己的变量一般不要用这种变量名。

类似`_xxx`和`__xxx`这样的函数或变量就是非公开的（private），不应该被直接引用，比如`_abc`，`__abc`等；

之所以我们说，private函数和变量“不应该”被直接引用，而不是“不能”被直接引用，是因为Python并没有一种方法可以完全限制访问private函数或变量，但是，从编程习惯上不应该引用private函数或变量。

private函数或变量不应该被别人引用，那它们有什么用呢？请看例子：

```python
def _private_1(name):
    return 'Hello, %s' % name

def _private_2(name):
    return 'Hi, %s' % name

def greeting(name):
    if len(name) > 3:
        return _private_1(name)
    else:
        return _private_2(name)
```

我们在模块里公开`greeting()`函数，而把内部逻辑用private函数隐藏起来了，这样，调用`greeting()`函数不用关心内部的private函数细节，这也是一种非常有用的代码封装和抽象的方法，即：

外部不需要引用的函数全部定义成private，只有外部需要引用的函数才定义为public。





## 面向对象编程（OOP）

OOP把对象作为基本单元，一个对象包含了数据和操作数据的函数。

OOP把计算机程序视为一系列命令的集合，而每个对象都可以接受其他对象发过来的消息，并处理这些消息，计算机程序的执行就是一系列消息在对象之间传递。

在Python中，所有的数据类型都可以视为对象，当然也可以自定义对象。自定义的对象数据类型就是面向对象中的类（Class）的概念。



### 类和实例

OOP最重要概念就是类（Class）和实例（Instance），必须牢记类是抽象的模板，实例是根据类创建出来的一个个具体的“对象”，每个对象都拥有相同的方法，但是各自的数据可能不同。

仍以Student类为例，在Python中，定义类是通过`class`关键字：

```python
class Student(object):
    pass
```

`calss`后面紧接着是类名，即`Student`类创建出`Student`实例，创建实例是通过类名+()实现的：

```python
>>> bart = Student()
>>> bart
<__main__.Student object at 0x10a67a590>
>>> Student
<class '__main__.Student'>
```

可以看到，变量`bart`指向的就是一个`Student`的实例，后面的`0x10a67a590`是内存地址，每个object的地址都不一样，而`Student`本身则是一个类。

可以自由地给一个实例变量绑定属性，比如，给实例`bart`绑定一个`name`属性：

```python
>>> bart.name = 'Bart Simpson'
>>> bart.name
'Bart Simpson'
```

由于类可以起到模板作用，因此可以在创建实例的时候，把一些我们任务必须绑定的属性强制填进去。通过定义一个特殊的`__init__`方法，创建实例的时候就把`name`，`score`等属性绑定上去：

```python
class Student(object):
    
    def __init__(self,name,score):
        self.name = name
        self.score = score
```

注意，`__init__`方法的第一个参数永远是`self`，表示创建的实例本身，因此，在`__init__`方法内部，就可以把各种属性绑定到`self`,因为`self`就指向创建的实例本身。

有了`__init__`方法，在创建实例的时候，就不能传入空的参数了，必须传入与`__init__`方法相匹配的参数，但是self不需要传，Python解释器会自己把实例变量传进去。

```python
>>>bart = Student('Bart Simpson',59)
>>>bart.name
'Bart Simpson'
>>>bart.score
59
```

和普通的函数相比，在类中定义的函数只有一点不同，就是第一个参数永远是实例变量`self`，并且，调用时不用传递该参数。除此之外，类的方法和普通函数没有什么区别，所有，你仍然可以用默认参数、可变参数、关键字参数、命名关键字参数。

#### 数据封装

OOP的一个重要特点就是数据封装。在上面的`Student`类中，每个实例就拥有各自的`name`和`score`这些数据。我们可以通过函数来访问这些数据，比如打印一个学生的成绩。但是`Student`实例本身就拥有这些数据，要访问这些数据，就没有必要从外面的函数去访问，可以直接在`Student`类的内部定义访问数据的函数，这样，就把“数据”给封装起来了。这些封装数据的函数是和`Student`类本身是关联起来的，我们称之为类的方法：

```python
class Student(object):
    
    def __init__(self,name,score):
        self.name = name
        self.score = score
    
    def print_score(self):
        print('%s : %s' % (self.name, self.score))
```

要定义一个方法，除了第一个参数是`self`以外，其他和普通函数一样。要调用另一个方法，只需要在实例变量上直接调用，除了`self`不用传递，其他参数正常传入：

```python
>>>bart.print_score()
Bart Simpson: 59
```



### 访问限制

在class内部，可以有属性和方法，而外部代码可以通过直接调用实例变量的方法来操作数据，这样，就隐藏了内部的复杂逻辑。

但是从`Student`类的定义来看，外部代码还是可以自由的修改一个实例的`name`、`scorre`属性。

如果要让内部属性不被外部访问，可以把属性名称前加两个下划线`__`，在python中，实例的变量名如果以`__`开头，就变成了一个私有变量（private），只有内部可以访问，外部不能访问，所以，我们把`Student`类改一改：

```python
class Student(object):
    def __init__(self,name,score):
        self.__name = name
        self.__score = score
        
    def print_score(self):
        prnint('%s %s' % (self.__name, self.__score))
```

改完后，对外部代码来说没什么变动，但以及无法从外部访问`实例变量.__name`和`实例变量.__score`了

但是如果外部代码要获取score怎么办？可以给`Student`类增加`set_score`方法：

```python
class Student(object):
    ...
    
    def set_score(self,score):
        self.__score = score
```

你也许会问，原先那种直接通过`bart.score = 99`也可以修改啊，为什么要定义一个方法大费周折？因为在方法中，可以对参数做检查，避免传入无效的参数：

```python
class Student(object):
    ...

    def set_score(self, score):
        if 0 <= score <= 100:
            self.__score = score
        else:
            raise ValueError('bad score')
```



而不能直接访问`__name`是因为Python解释器对外把`__name`变量改成了`_Student__name`，所以，仍然可以通过`_Student__name`来访问`__name`变量：

```python
>>> bart._Student__name
'Bart Simpson'
```

但是强烈建议你不要这么干，因为不同版本的Python解释器可能会把`__name`改成不同的变量名。



```python
>>> bart = Student('Bart Simpson', 59)
>>> bart.get_name()
'Bart Simpson'
>>> bart.__name = 'New Name' # 设置__name变量！
>>> bart.__name
'New Name'
```

表面上看，外部代码“成功”地设置了`__name`变量，但实际上这个`__name`变量和class内部的`__name`变量*不是*一个变量！内部的`__name`变量已经被Python解释器自动改成了`_Student__name`，而外部代码给`bart`新增了一个`__name`变量。不信试试：

```python
>>> bart.get_name() # get_name()内部返回self.__name
'Bart Simpson'
```





### 继承和多态

在OOP程序设计中，当我们定义了一个class的时候，可以从某个现有的class继承，新的class称为子类（Subclass），而被继承的class称为基类，父类或超类（Base class，super class）。

比如我们编写了一个名为`Animal`的class，有一个`run`方法可以直接打印：

```python
class Animal(object):
    def run(self):
        print('Animal is running……')
```

当我们要编写dog和cat类时，就可以直接从`Animal`类继承：

```python
class Dog(Animal):
    pass

class Cat(Animal):
    pass
```

对于`Dog`类来说，`Animal`就是它的父类，对于`Animal`类来说，`Dog`就是它的子类。`Cat`和`Dog`类似。

继承有什么好处?最大的好处是，子类获得了父类的全部功能。由于`Animal`实现了`run()`方法，因此，`Dog`和`Cat`作为它的子类，什么事没干就自动拥有了`run()`方法：

```python
dog = Dog()
dog.run

#cat与dog的处理方式一致
cat = Cat()
cat.run
```

运行结果如下

```python
Animal is running...
Animal is running...
```



当然，也可以对子类增加一些方法，比如Dog类：

```python
class Dog(Animal):

    def run(self):
        print('Dog is running...')

    def eat(self):
        print('Eating meat...')
```



继承的第二个好处需要我们对代码做一点改进。你看到了，无论是`Dog`还是`Cat`，它们`run()`的时候，显示的都是`Animal is running...`，符合逻辑的做法是分别显示`Dog is running...`和`Cat is running...`，因此，对`Dog`和`Cat`类改进如下：

```python
class Dog(Animal):

    def run(self):
        print('Dog is running...')

class Cat(Animal):

    def run(self):
        print('Cat is running...')
```



再次运行结果如下：

```python
Dog is running...
Cat is running...
```

当子类和父类都存在相同的`run()`方法时，我们说，子类的`run`覆盖了父类的`run`，在代码运行的时候，总是会调用子类的`run`。这样我们就获得了继承的另一个好处：多态。

要理解什么是多态，我们首先要对数据类型再作一点说明。当我们定义一个class的时候，我们实际上就定义了一种数据类型。

判断一个变量是否是某个类型可以用`isinstance()`判断：

```python
>>> isinstance(a, list)
True
>>> isinstance(b, Animal)
True
>>> isinstance(c, Dog)
True
```

看来`a`、`b`、`c`确实对应着`list`、`Animal`、`Dog`这3种类型。

但是等等，试试：

```Python
>>> isinstance(c, Animal)
True
```

看来`c`不仅仅是`Dog`，`c`还是`Animal`！

在继承关系中，如果一个数据类型是某个数据类型的子类，那它的数据类型也可以被看成是父类。但是反过来就不行：

```python
>>>b = Animal()
>>>isinstance(b,Dog)
False
```



理解多态的好处，我们还需要编写一个函数，接受一个`Anmial`类型的变量：

```python
def run_twice(animal):
    animal.run()
    animal.run()
```

当我们传入`Animal`的实例时，`run_twice()`就打印出：

```python
>>> run_twice(Animal())
Animal is running...
Animal is running...
```

传入`Dog`的实例时，`run_twice()`打印：

```python
>>>run_twice(Dog())
Dog is runninng……
Dog is runninng……
```

现在我们再定义一个`Tortoise`类型，也从`Animal`派生：

```python
class Tortoise(Animal):
    def run(self)
    	print('Tortoise is running slowwly…')
```

当我们调用`run_twice()`时，传入`Tortoise`的实例：

```python
>>> run_twice(Tortoise())
Tortoise is running slowly...
Tortoise is running slowly...
```

你会发现，新增一个`Animal`的子类，不必对`run_twice`做任何修改，实际上，任何依赖`Animal`作为参数的函数或者方法都可以不加修改的正常运行，原因就在于多态。

多态的好处是，当我们传入`Dog`，`Cat`，`Tortoise`……时，我们只需要接受`Anmial`类型就可以了，因为`Dog`，`Cat`，`Tortoise`……都是`Animal`类型，然后，按照`Animal`类型进行操作即可。由于`Anmial`类型有`run()`方法，因此，传入的任意类型，只要是`Aniaml`类或者是子类，就会自动调用实际类型的`run()`方法，这就是多态的意思。

对于一个变量，我们只需要知道它是`Animal`类型，无需确切地知道它的子类型，就可以放心地调用`run()`方法，而具体调用的`run()`方法是作用在`Animal`、`Dog`、`Cat`还是`Tortoise`对象上，由运行时该对象的确切类型决定，这就是多态真正的威力：调用方只管调用，不管细节，而当我们新增一种`Animal`的子类时，只要确保`run()`方法编写正确，不用管原来的代码是如何调用的。这就是著名的“开闭”原则：

对扩展开放：允许新增`Animal`子类；

对修改封闭：不需要修改依赖`Animal`类型的`run_twice()`等函数。



### 获取对象信息

当我们拿到一个对象引用时，如何知道这个对象是什么类型，有那些方法呢？

#### 使用`type()`

我们判断对象类型，使用`type`函数，基本类型都可以用`type`判断：

```python
>>> type(123)
<class 'int'>
```

如果一个对象指向函数或者类，也可以用`type()`判断。但是`type()`函数返回的是什么类型呢？它返回对应的Class类型。



如果要判断一个对象是否是函数怎么办？可以使用`types`模块中定义的常量：

```python
>>> import types
>>> def fn():
...     pass
...
>>> type(fn)==types.FunctionType
True
>>> type(abs)==types.BuiltinFunctionType
True
>>> type(lambda x: x)==types.LambdaType
True
>>> type((x for x in range(10)))==types.GeneratorType
True
```

#### 使用dir()

如果要获得一个对象的所有属性和方法，可以使用`dir()`函数，它返回一个包含字符串的list，比如，获得一个str对象的所有属性和方法：

```python
>>> dir('ABC')
['__add__', '__class__',..., '__subclasshook__', 'capitalize', 'casefold',..., 'zfill']
```

紧接着，可以测试对象的属性：

```python
>>>hasattr(obj,'x') #有x属性吗？
>>>setattr(obj,'y') #设置x属性
>>> getattr(obj, 'x') # 获取属性'x'
```

可以传入一个default参数，如果属性不存在，就返回默认值：

```python
>>> getattr(obj, 'z', 404) # 获取属性'z'，如果不存在，返回默认值404
404
```

也可以获得对象的方法：

```python
>>> hasattr(obj, 'power') # 有属性'power'吗？
True
>>> getattr(obj, 'power') # 获取属性'power'
<bound method MyObject.power of <__main__.MyObject object at 0x10077a6a0>>
```



## 面向对象高级编程

### 使用\_\_slots\_\_

正常情况下，当我们定义了一个class，创建了一个clsaa的实例后，我们可以给该实例绑定任何实例和方法，这就是动态语言的灵活性。先定义class：

```python
class Student(object):
    pass
```

然后，尝试给实例绑定一个属性：

```python
>>>s = Student
>>>s.name = 'Michael' #动态给实例绑定一个对象
>>>print('s.name')
Micheal
```

还可以尝试给实例绑定一个方法：

```python
>>>def set_age(self,age): #定义一个函数作为实例方法
    	self.age=age
    
>>>from type import MethodType
>>>s.set_age = MethodeType(set_age, s) #给实例绑定一个方法
>>>s.set_age(25) #调用实例方法
>>>s.age #测试结果
25
```

但是==给一个实例绑定的方法对另一个实例是不起作用的。==





## 进程和线程

### 多进程

Unix/Linux操作系统提供了一个`fork`调用，普通的函数调用，调用一次返回一次，但是`fork`调用一次，返回两次，因为操作系统自动把当前进程（父进程）复制了一份（称为子进程），然后分别在父进程和子进程内返回。

子进程永远返回`0`，而父进程返回子进程的ID，子进程只需要调用`getppid()`就可以拿到父进程的ID。

Python的`os`模块封装了常见的系统调用，其中就包括`fork`，可以在Python程序中轻松创建子进程：

```python
import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
```

如果要启动大量子进程，可以用进程池的方法批量创建子进程：

```python
from multiprocessing import Pool
import os, timme, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))
    
if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
```

代码解读：

对`Pool`对象调用`join()`方法会等待所有子进程执行完毕，调用`join()`之前必须先调用`close()`，调用`close()`之后就不能继续添加新的`Process`了。



例子：

```python
pool = Pool(8) #可以同时跑8个进程
	pool.map(get_all, [i for i in range(10)])
    pool.close
    pool.join()
```

这里的`pool.close()`是说关闭pool，使其不在接受新的（主进程）任务。

这里的`pool.join()`是说：主进程阻塞后，让子进程继续运行完成，子进程运行完成后，再把主进程全部关掉。



#### 子进程

很多时候，子进程并不是自生，而是一个外部进程。我们创建了子进程后，还需要控制子进程的输入和输出。

==subprocess==模块可以让我们非常方便的启动一个子进程，然后控制其输入和输出。

下面的例子演示了如何在Python代码中运行命令`nslookup www.python.org`，这和命令行的效果是一样的：

```python
import subprocess

print('$ nslookup www.python.org')
r = subprocess.call(['nslookup','www.python.org'])
print('Exit code:', r)
```

```python
$ nslookup www.python.org
Server:		192.168.19.4
Address:	192.168.19.4#53

Non-authoritative answer:
www.python.org	canonical name = python.map.fastly.net.
Name:	python.map.fastly.net
Address: 199.27.79.223

Exit code: 0
```



#### subprocess模块详解

`subprocess`模块是**Python 2.4**中新增的一个模块，它允许你生成新的进程，连接到它们的**input/output/error**管道，并获取它们的返回（状态）码。这个模块的目的在于替换几个旧的模块和方法，如：

```python
os.system
os.spawn*
```



**1.subprocess模块中的常用函数**

| 函数                            | 描述                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| subprocess.run()                | py3.5中新增的函数。执行制定的命令，等待命令执行完毕后返回一个包含执行结果的CompletedProcess类的实例。 |
| subprocess.call()               | 执行指定命令，返回命令执行状态，其功能类似于`os.system(cmd)` |
| subprocess.check_call()         | python2.5中新增的函数，执行制定的命令，如果执行成功则返回状态码，否则抛出异常。其功能等价于subprocess.run(..., check = True) |
| subprocess.check_output()       | python 2.7中新增的函数。执行制定的命令，如果执行状态码为0则返回命令执行结果。都则抛出异常。 |
| subprocess.getoutput(cmd)       | 接受字符串格式的命令，执行命令并返回执行结果，其功能类似于os.popen(cmd).reead()和commands.getoutput(cmd) |
| subprocess.getstatusoutput(cmd) | 执行cmd命令，返回一个元组（命令执行状态，命令执行结果输出），其功能类似于commands,getstatusoutput() |



> 说明：
>
> 在Python 3.5之后的版本中，官方文档中提倡通过subprocess.run()函数替代其他函数来使用subproccess模块的功能；
> 在Python 3.5之前的版本中，我们可以通过subprocess.call()，subprocess.getoutput()等上面列出的其他函数来使用subprocess模块的功能；
> subprocess.run()、subprocess.call()、subprocess.check_call()和subprocess.check_output()都是通过对subprocess.Popen的封装来实现的高级函数，因此如果我们需要更复杂功能时，可以通过subprocess.Popen来完成。
> subprocess.getoutput()和subprocess.getstatusoutput()函数是来自Python 2.x的commands模块的两个遗留函数。它们隐式的调用系统shell，并且不保证其他函数所具有的安全性和异常处理的一致性。另外，它们从Python 3.3.4开始才支持Windows平台。



**上面各函数的定义以及参数说明**

函数参数列表

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, shell=False, timeout=None, check=False, universal_newlines=False)

subprocess.call(args, *, stdin=None, stdout=None, stderr=None, shell=False, timeout=None)

subprocess.check_call(args, *, stdin=None, stdout=None, stderr=None, shell=False, timeout=None)

subprocess.check_output(args, *, stdin=None, stderr=None, shell=False, universal_newlines=False, timeout=None)

subprocess.getstatusoutput(cmd)

subprocess.getoutput(cmd)
```

参数说明：

- args：要执行的shell命令，默认应该是一个字符串序列，如['df', '-Th']，也可以是一个字符串，但要把shell的参数的值设置为True

- shell：如果sehll为True，那么指定的命令将通过shell执行。

- check：如果check参数的值是True，且执行命令的进程以非0状态码退出，则会抛出一个CalledProcessError的异常，且该异常会包含参数、退出状态码、以及stdout和stderr

- `stdout, stderr：input`： 该参数是传递给Popen.communicate()，通常该参数的值必须是一个字节序列，如果universal_newlines=True，则其值应该是一个字符串。

  1. run()函数默认不会捕获命令执行结果的正常输出和错误输出，如果我们向获取这些内容需要传递subprocess.PIPE，然后可以通过返回的CompletedProcess类实例的stdout和stderr属性或捕获相应的内容；

  2. call()和check_call()函数返回的是命令执行的状态码，而不是CompletedProcess类实例，所以对于它们而言，stdout和stderr不适合赋值为subprocess.PIPE；
     3
  3. check_output()函数默认就会返回命令执行结果，所以不用设置stdout的值，如果我们希望在结果中捕获错误信息，可以执行stderr=subprocess.STDOUT。

- `universal_newlines`： 该参数影响的是输入与输出的数据格式，比如它的值默认为False，此时stdout和stderr的输出是字节序列；当该参数的值设置为True时，stdout和stderr的输出是字符串。



**3.subprocess.CompletedProcess类介绍**

需要说明的是，`subprocess.run()`函数是Python3.5中新增一个高级函数，其返回值是一个`subprocess.CompletedProcess`类的实例，因此，subprocess,completedProcess类也是Python 3.5中才存在的。它表示的是一个以结束进程的状态信息。



#### subprocess.Popen介绍

该类用于在一个新的程序中执行一个子程序。前面我们提到过，上面介绍的这些函数艘是基于`subprocess.Popen`类实现的，通过使用这些被封装的高级函数可以很方便的完成一些常见需求。由于`subprocess`模块底层的进程创建和管理是有Popen类来处理的，因此，当我们无法通过上面哪些高级函数来实现一些不太常见的功能时就可以通过subprocess.Popen类提供灵活的api来完成。

1.subprocess.Popen构造函数

```python
class subprocess.Popen(args, bufsize=-1, executable=None, stdin=None, stdout=None, stderr=None, 
    preexec_fn=None, close_fds=True, shell=False, cwd=None, env=None, universal_newlines=False,
    startup_info=None, creationflags=0, restore_signals=True, start_new_session=False, pass_fds=())
```

- args： 要执行的shell命令，可以是字符串，也可以是命令各个参数组成的序列。当该参数的值是一个字符串时，该命令的解释过程是与平台相关的，因此通常建议将args参数作为一个序列传递。
- bufsize： 指定缓存策略，0表示不缓冲，1表示行缓冲，其他大于1的数字表示缓冲区大小，负数 表示使用系统默认缓冲策略。
- stdin, stdout, stderr： 分别表示程序标准输入、输出、错误句柄。
- preexec_fn： 用于指定一个将在子进程运行之前被调用的可执行对象，只在Unix平台下有效。
- close_fds： 如果该参数的值为True，则除了0,1和2之外的所有文件描述符都将会在子进程执行之前被关闭。
- shell： 该参数用于标识是否使用shell作为要执行的程序，如果shell值为True，则建议将args参数作为一个字符串传递而不要作为一个序列传递。
- cwd： 如果该参数值不是None，则该函数将会在执行这个子进程之前改变当前工作目录。
- env： 用于指定子进程的环境变量，如果env=None，那么子进程的环境变量将从父进程中继承。如果env!=None，它的值必须是一个映射对象。
- universal_newlines： 如果该参数值为True，则该文件对象的stdin，stdout和stderr将会作为文本流被打开，否则他们将会被作为二进制流被打开。
- startupinfo和creationflags： 这两个参数只在Windows下有效，它们将被传递给底层的CreateProcess()函数，用于设置子进程的一些属性，如主窗口的外观，进程优先级等。



2. subprocess.Popen类的实例可调用的方法

| 方法                                        | 描述                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| Popen.poll()                                | 用于检查子进程（命令）是否已经执行结束，没结束返回None，结束后返回状态码。 |
| Popen.wait(timeout=None)                    | 等待子进程结束，并返回状态码；如果在timeout指定的秒数之后进程还没有结束，将会抛出一个TimeoutExpired异常。 |
| Popen.communicate(imput=None, timeout=None) | 该方法可用于来与程序进行交互，比如发送数据到stdin，从stdout和stderr读取数据，直到达到文件末尾。 |
| Popen.send_signal(signal)                   | 发送指定的信号给这个子进程                                   |
| Popen.terminate()                           | 停止这个子进程                                               |
| Popen.kill                                  | 杀死该子进程                                                 |





#### 进程间通信

`Process`之间肯定是需要通信的，操作系统提供了很多的机制来实现进程间的通信。Python的`nultiprocessing`模块包装了底层的机制，提供了`Queue`，`Pipes`等多种方式来交换数据。

我们以`Queue`为例，在父进程中创建两个子进程，一个往`Queue`里写入数据，一个从`Queue`里读取数据：

```python
from multiprocessing import Process, Queue
import os, time, rendom

#写数据进程执行的代码
def write(q):
    print('Pricess to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

#读数据进程执行的代码
def read(q):
    print('Process to readL %s' % os.getpid)
	write True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__ = '__main__':
    #父进程创建的Queue，并传给各个子进程
    q = Queue()
    pw = Process(targe=write, args=(q,))
    pr = Process(target=read, args=(q,))
    #启动子进程pr,读取：
    pr.start()
    #等待pw结束
    pw.join()
    #pr进程里是四循环，无法等待其结束，只能强行终止：
    pr.terminate()
```

---------





### 多线程

进程是由若干个线程组成的，一个进程至少有一个线程。

由于线程的操作系统直接支持的执行单元，因此，高级语言通常都内置多线程的支持，Python也不例外，并且，Python的线程是真正的Posix Thread，而不是模拟出来的线程。

Python的标准库提供了两个模块：`_thread`和`threading`，`_thread`是低级模块，`threading`是高级模块，对`_thread`进行了封装。绝大多数情况下，我们只需要使用`_threading`这个高级模块。

启动一个线程就是把一个函数传入并创建`Thread`实例，然后调用`start()`开始执行：

```python
import time, threading

#新线程执行的代码：
def loop:
    print('thread %s is runing...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n+ 1
        print('thread %s >>> %s' % (threading.current_thread ().name))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)
    
print('thread %s is running...' % threading.current_thread().name)
t = threading.Thread(target=loop, name = 'LoopThread')
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)
```

由于任何进程默认就会启动一个线程，我们把该线程称为主线程，主线程又可以启动新的线程，Python的`threading`模块有个`current_thread()`函数，它永远返回当前线程的实例。主线程实例的名字叫`MainThread`，子线程的名字在创建时指定，我们用`LoopThread`命名子线程。名字仅仅在打印时用来显示，完全没有其他意义，如果不起名字Python就自动给线程命名为`Thread-1`，`Thread-2`……



#### Lock

多线程和多进程最大的不同在于，多进程中，同一个变量，各自有一份拷贝存在于每个进程中，互不影响，而多线程中，所有的变量都由所有的线程共享，所以，任何一个变量都可以被任何一个线程修改，因此，线程之间共享数据最大的危险在于多个线程同时修改一个变量，把内容给该乱了。

所以我们希望创建一把锁，当某个程序开始执行一个线程时，我们说，该线程获得的锁，因此其他线程不能同时执行该线程，只能等待，，直到锁被释放后，获得该锁以后才能改。由于锁只有一个，无论多少个线程，同一时刻最多只有一个线程持有该锁，所以不会造成修改冲突。创建一个锁就是通过`threading.Lock()`来实现：

```python
balance = 0
lock = threading.Lock()

def run_thread(n):
    for i in range(10000)
    #先要获取锁：
    lock.acquire()
    try:
        change_it(n)
    finally:
        #改完了一定要释放锁
        lock.release()
```

当多个线程同时执行`lock.acquire()`时，只有一个线程能成功地获取锁，然后继续执行代码，其他线程就继续等待直到获得锁为止。

获得锁的线程用完后一定要释放锁，否则那些苦苦等待锁的线程将永远等待下去，成为死线程。所以我们用`try...finally`来确保锁一定会被释放。

锁的好处就是确保了某段关键代码只能由一个线程从头到尾完整地执行，坏处当然也很多，首先是阻止了多线程并发执行，包含锁的某段代码实际上只能以单线程模式执行，效率就大大地下降了。其次，由于可以存在多个锁，不同的线程持有不同的锁，并试图获取对方持有的锁时，可能会造成死锁，导致多个线程全部挂起，既不能执行，也无法结束，只能靠操作系统强制终止。



#### 多核CPU

如果你不幸拥有一个多核CPU，你肯定在想，多核应该可以同时执行多个线程。

如果写一个死循环的话，会出现什么情况呢？

打开Mac OS X的Activity Monitor，或者Windows的Task Manager，都可以监控某个进程的CPU使用率。

我们可以监控到一个死循环线程会100%占用一个CPU。

如果有两个死循环线程，在多核CPU中，可以监控到会占用200%的CPU，也就是占用两个CPU核心。

要想把N核CPU的核心全部跑满，就必须启动N个死循环线程。

试试用Python写个死循环：

```python
import threading, multiprocessing

def loop():
    x = 0
    while True:
        x = x ^ 1

for i in range(multiprocessing.cpu_count()):
    t = threading.Thread(target=loop)
    t.start()
```

启动与CPU核心数量相同的N个线程，在4核CPU上可以监控到CPU占用率仅有102%，也就是仅使用了一核。

但是用C、C++或Java来改写相同的死循环，直接可以把全部核心跑满，4核就跑到400%，8核就跑到800%，为什么Python不行呢？

因为Python的线程虽然是真正的线程，但解释器执行代码时，有一个GIL锁：Global Interpreter Lock，任何Python线程执行前，必须先获得GIL锁，然后，每执行100条字节码，解释器就自动释放GIL锁，让别的线程有机会执行。这个GIL全局锁实际上把所有线程的执行代码都给上了锁，所以，多线程在Python中只能交替执行，即使100个线程跑在100核CPU上，也只能用到1个核。

GIL是Python解释器设计的历史遗留问题，通常我们用的解释器是官方实现的CPython，要真正利用多核，除非重写一个不带GIL的解释器。

所以，在Python中，可以使用多线程，但不要指望能有效利用多核。如果一定要通过多线程利用多核，那只能通过C扩展来实现，不过这样就失去了Python简单易用的特点。

不过，也不用过于担心，Python虽然不能利用多线程实现多核任务，但可以通过多进程实现多核任务。多个Python进程有各自独立的GIL锁，互不影响。
