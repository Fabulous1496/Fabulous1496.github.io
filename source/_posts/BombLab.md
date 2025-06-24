---
title: BombLab
date: 2024-07-07 11:11:49
categories:
- CMU15-213
index_img: /Pictures/CMU15-213/bomblab.jpg
banner_img: /Pictures/CMU15-213/bomblab.jpg
---

# BombLab

**Lab概述**：“二进制炸弹”是一个 Linux 可执行的 C 程序，包含六个“阶段”。每个阶段期望学生在标准输入上输入特定字符串。如果学生输入了预期的字符串，则该阶段被“拆除”。否则炸弹会通过打印“BOOM!!!”来“爆炸”。学生的目标是尽可能多地“拆除”该炸弹。

因此，本Lab聚焦于利用gdb等调试器来反编译二进制文件，并逐步执行每个阶段的机器代码，利用学习的汇编知识来拆解隐藏在代码中的字符串密码。

## 预备知识：GDB调试器的使用

GNU的调试器GDB提供了许多有用的特性，支持机器及程序的运行时评估和分析。我们使用下面指令表中的命令来使用GDB：

| 命令                    | 效果                                    |
| ----------------------- | --------------------------------------- |
| gdb fileName            | 开始调试fileName文件                    |
| run 1 2 3               | 运行程序（在此给出命令行参数）          |
| kill                    | 终止程序                                |
| quit                    | 退出GDB                                 |
| break func              | 在函数func入口处设置断点                |
| break * 0x10000         | 在地址 0x10000 处设置断点               |
| delete 1                | 删除断点1，无index默认删除所有          |
| stepi                   | 执行1条指令（后跟数字表示运行几条指令） |
| nexti                   | 类似于stepi，以函数为调用单位           |
| continue                | 继续执行                                |
| disas                   | 反汇编当前函数                          |
| disas func              | 反汇编函数func                          |
| disas 0x10000           | 反汇编地址0x10000附近的函数             |
| disas 0x10000 0x100FF   | 反汇编地址范围内的代码                  |
| print /x $rip           | 以十六进制输出程序计数器的值            |
| print $rax              | 以十进制输出%rax的内容                  |
| print /x $rip           | 以十六进制输出%rax的内容                |
| print /t $rax           | 以二进制输出%rax的内容                  |
| print /x 555            | 输出555的十六进制表示                   |
| print /x ($rsp+8)       | 以十六进制输出%rsp的内容加8             |
| print *(int \*) 0x10000 | 将0x10000地址所存储的内容以数字形式输出 |
| print (char *) 0x10000  | 输出存储在0x10000的字符串               |
| x/ $rsp                 | 解析%rsp中的内容                        |
| x/w $rsp                | 解析在%rsp所指位置的word（8字节）       |
| info registers          | 寄存器信息                              |
| info stack              | 栈信息                                  |
| info functions          | 函数信息                                |

更多指令可以查询文档或使用/help指令。

## 正式阶段：开始拆弹

### 原码总览

抛去营造氛围感和背景故事的“Doctor Evil”的注释，整个bomb.c文件的结构如下所示：

```c
#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#include "phases.h"

FILE *infile;

int main(int argc, char *argv[])
{
    char *input;

    if (argc == 1) {  
	infile = stdin;
    } 

    else if (argc == 2) {
	if (!(infile = fopen(argv[1], "r"))) {
	    printf("%s: Error: Couldn't open %s\n", argv[0], argv[1]);
	    exit(8);
	}
    }

    /* You can't call the bomb with more than 1 command line argument. */
    else {
	printf("Usage: %s [<input_file>]\n", argv[0]);
	exit(8);
    }

    initialize_bomb();

    printf("Welcome to my fiendish little bomb. You have 6 phases with\n");
    printf("which to blow yourself up. Have a nice day!\n");

    input = read_line();             
    phase_1(input);                  
    phase_defused();                 
    printf("Phase 1 defused. How about the next one?\n");

    input = read_line();
    phase_2(input);
    phase_defused();
    printf("That's number 2.  Keep going!\n");


    input = read_line();
    phase_3(input);
    phase_defused();
    printf("Halfway there!\n");

    input = read_line();
    phase_4(input);
    phase_defused();
    printf("So you got that one.  Try this one.\n");

    input = read_line();
    phase_5(input);
    phase_defused();
    printf("Good work!  On to the next...\n");

    input = read_line();
    phase_6(input);
    phase_defused();
    
    return 0;
}

```

整段代码的逻辑很简单，首先定义了一个File文件infile作为我们输入密码的文件，然后代码会从infile中读取我们的密码完成各个阶段的解密。而解密所需要的字符就储存在二进制文件bomb中。

使用指令`objdump -d bomb > bomb.asm`，我们便得到了反编译后的文件。

### Phase_1

阅读文件，我们找到phase_1对应的代码：

```assembly
0000000000400ee0 <phase_1>:
  400ee0:	48 83 ec 08          	sub    $0x8,%rsp
  400ee4:	be 00 24 40 00       	mov    $0x402400,%esi
  400ee9:	e8 4a 04 00 00       	call   401338 <strings_not_equal>
  400eee:	85 c0                	test   %eax,%eax
  400ef0:	74 05                	je     400ef7 <phase_1+0x17>
  400ef2:	e8 43 05 00 00       	call   40143a <explode_bomb>
  400ef7:	48 83 c4 08          	add    $0x8,%rsp
  400efb:	c3                   	ret    
```

我们发现，其中包含一个检查字符串是否相等的的函数`strings_not_equal`，如果相等，则跳转到地址0x400ef7处（add行），如果不相等，则调用`explode_bomb`，不难想象会发生什么。于是我们首先就是要在`explode_bomb`函数处打一个断点，确保我们接下来的操作不会引起炸弹爆炸。

分析各个寄存器，我们发现出现了%rsp，%esi，%eax三个寄存器。其中%rsp为栈指针，指明运行时栈的栈顶位置，而%esi为%rsi的低8字节，保存函数调用的第二个参数，%eax作为%rax的低8字节，保存函数的返回值。所以我们就刻画出了函数`strings_not_equal`的基本形式：接受至少两个参数并返回一个值。不难想到第一个参数就是我们输入的字符串，第二个参数存放在地址**0x402400**的字符串就是我们的密码。

接下来我们来操作一下，首先打断点并运行：

```bash
gdb bomb
(gdb) break explode_bomb
(gdb) break phase_1
(gdb) run
```

这里会让你输入密码，可以先随便输入一个到达断点。

之后我们便到达了断点phase_1，用`disas`指令看看它的~~源~~码：

```bash
Breakpoint 2, 0x0000000000400ee0 in phase_1 ()
(gdb) disas
Dump of assembler code for function phase_1:
=> 0x0000000000400ee0 <+0>:     sub    $0x8,%rsp
   0x0000000000400ee4 <+4>:     mov    $0x402400,%esi
   0x0000000000400ee9 <+9>:     call   0x401338 <strings_not_equal>
   0x0000000000400eee <+14>:    test   %eax,%eax
   0x0000000000400ef0 <+16>:    je     0x400ef7 <phase_1+23>
   0x0000000000400ef2 <+18>:    call   0x40143a <explode_bomb>
   0x0000000000400ef7 <+23>:    add    $0x8,%rsp
   0x0000000000400efb <+27>:    ret
```

可以看到和我们上面解析出来的代码一致。

继续使用`stepi`指令向下执行到mov指令结束：

```
(gdb) stepi 2
(gdb) disas
Dump of assembler code for function phase_1:
   0x0000000000400ee0 <+0>:     sub    $0x8,%rsp
   0x0000000000400ee4 <+4>:     mov    $0x402400,%esi
=> 0x0000000000400ee9 <+9>:     call   0x401338 <strings_not_equal>
   0x0000000000400eee <+14>:    test   %eax,%eax
   0x0000000000400ef0 <+16>:    je     0x400ef7 <phase_1+23>
   0x0000000000400ef2 <+18>:    call   0x40143a <explode_bomb>
   0x0000000000400ef7 <+23>:    add    $0x8,%rsp
   0x0000000000400efb <+27>:    ret
End of assembler dump.
```

好的，此时地址**0x402400**处的字符串已经成功移到%esi寄存器中了。我们使用`x/s $esi`可以得到内容字符串：

```
(gdb) x/s $esi
0x402400:       "Border relations with Canada have never been better."
```

这便是我们第一阶段的密钥。

由于这里我们已经得到了字符串存放的源地址，更简单的方法是直接解析对应地址，不需要利用寄存器。但由于实际上每次运行代码的地址是不确定的，一般也不会直接告知地址，所以我采取了一种迂回的方法。

```bash
(gdb) print (char*) 0x402400
$1 = 0x402400 "Border relations with Canada have never been better."
```

这样也行。

### Phase_2

我们首先检验phase_1的密码是否正确。

```bash
(gdb) break phase_2
Breakpoint 1 at 0x400efc
(gdb) break explode_bomb
Breakpoint 2 at 0x40143a
(gdb) run
Welcome to my fiendish little bomb. You have 6 phases with
which to blow yourself up. Have a nice day!
Border relations with Canada have never been better.
Phase 1 defused. How about the next one?
aaa

Breakpoint 1, 0x0000000000400efc in phase_2 ()
```

可以看到，断点成功停在了phase_2处，说明我们成功通过阶段1.接下来把phase_2的代码反汇编出来：

```bash
(gdb) disas
Dump of assembler code for function phase_2:
=> 0x0000000000400efc <+0>:     push   %rbp
   0x0000000000400efd <+1>:     push   %rbx
   0x0000000000400efe <+2>:     sub    $0x28,%rsp
   0x0000000000400f02 <+6>:     mov    %rsp,%rsi
   0x0000000000400f05 <+9>:     call   0x40145c <read_six_numbers>
   0x0000000000400f0a <+14>:    cmpl   $0x1,(%rsp)
   0x0000000000400f0e <+18>:    je     0x400f30 <phase_2+52>
   0x0000000000400f10 <+20>:    call   0x40143a <explode_bomb>
   0x0000000000400f15 <+25>:    jmp    0x400f30 <phase_2+52>
   0x0000000000400f17 <+27>:    mov    -0x4(%rbx),%eax
   0x0000000000400f1a <+30>:    add    %eax,%eax
   0x0000000000400f1c <+32>:    cmp    %eax,(%rbx)
   0x0000000000400f1e <+34>:    je     0x400f25 <phase_2+41>
   0x0000000000400f20 <+36>:    call   0x40143a <explode_bomb>
   0x0000000000400f25 <+41>:    add    $0x4,%rbx
   0x0000000000400f29 <+45>:    cmp    %rbp,%rbx
   0x0000000000400f2c <+48>:    jne    0x400f17 <phase_2+27>
   0x0000000000400f2e <+50>:    jmp    0x400f3c <phase_2+64>
   0x0000000000400f30 <+52>:    lea    0x4(%rsp),%rbx
   0x0000000000400f35 <+57>:    lea    0x18(%rsp),%rbp
   0x0000000000400f3a <+62>:    jmp    0x400f17 <phase_2+27>
   0x0000000000400f3c <+64>:    add    $0x28,%rsp
   0x0000000000400f40 <+68>:    pop    %rbx
   0x0000000000400f41 <+69>:    pop    %rbp
   0x0000000000400f42 <+70>:    ret
End of assembler dump.
```

可以发现内部的主体存在函数`read_six_numbers`以及引爆程序`explode_bomb`。不难猜出函数`read_six_numbers`便是检验我们答案的函数，答案包括6个数字。

来逐步分析这段代码的逻辑：

1. 将%rbp和%rbx压入栈中，再申请一个0x28（40字节）的空间
2. 将栈顶%rsp存放的数据转移到%rsi中作为函数`read_six_numbers`的第二个参数，调用函数。
3. 比较（%rsp）和1，如果不相等，炸弹爆炸。
4. 如果相等，跳转到0x400f30行（+52  lea行），这行将%rsp+0x4的值存入%rbx中，之后将%rsp+0x18的值存入%rbp。
5. 再跳转到0x400f17（+27  mov行），将%rbx-0x4的值存入%eax，再将%eax中的值翻倍。之后比较（%rbx）（这是个间接取址）和%eax的值，如果不等，炸弹爆炸。
6. 如果相等，跳转到0x400f25行（+41  add行），将%rbx的值转变为%rbx+0x4，再比较%rbx和%rbp的值。
7. 如果不相等，跳转到0x400f17（+27  mov行），形成循环
8. 如果相等，跳转到0x400f3c（+64  add行），将%rsp+0x28（减小40字节），%rbx，%rbp出栈，退出程序。

我们尝试翻译为C语言代码：这里由于数据类型为int，故地址操作%rsp+4转换为指针便为rsp+1.其它地方也进行了相同的处理。

```c
int phase2(int *rsp) 
{
    int *rbx, *rbp, eax；
    if(*rsp == 1)
    {
        goto L2;
    }
    else
    {
        explode_bomb();
    }
L2:
    rbx = rsp+1;
    rbp = rsp+6;
    goto L3;
L3:
    eax = *(rbx-1);
    eax *= 2;
    if(eax == *rbx)
    {
        goto L4;
    }
    else
    {
        explode_bomb();
    }
L4:
    rbx = rbx+1;
    if(rbx != rbp)
    {
        goto L3;
    }
    else
    {
        return ;
    }
        	
}
```

我们可以继续查看关键函数`read_six_numbers`的代码：

```assembly
=> 0x0000000000400efc <+0>:     push   %rbp
   0x0000000000400efd <+1>:     push   %rbx
   0x0000000000400efe <+2>:     sub    $0x28,%rsp
   0x0000000000400f02 <+6>:     mov    %rsp,%rsi
   0x0000000000400f05 <+9>:     call   0x40145c <read_six_numbers>
   # 下面注意%rsi=%rsp
Dump of assembler code for function read_six_numbers:
=> 0x000000000040145c <+0>:     sub    $0x18,%rsp  	#申请空间
   0x0000000000401460 <+4>:     mov    %rsi,%rdx	#将%rsi赋值给%rdx
   0x0000000000401463 <+7>:     lea    0x4(%rsi),%rcx	#将%rsi+4的地址存入%rcx
   0x0000000000401467 <+11>:    lea    0x14(%rsi),%rax	#将%rsi+20的地址存入%rax
   0x000000000040146b <+15>:    mov    %rax,0x8(%rsp)	#将%rax赋值给%rsp+8
   0x0000000000401470 <+20>:    lea    0x10(%rsi),%rax	#将%rsi+16的地址存入%rax
   0x0000000000401474 <+24>:    mov    %rax,(%rsp)	#将rax的值赋给%rsp指向的地址
   0x0000000000401478 <+28>:    lea    0xc(%rsi),%r9#将%rsi+12存入%r9
   0x000000000040147c <+32>:    lea    0x8(%rsi),%r8#将%rsi+8存入%r8
   0x0000000000401480 <+36>:    mov    $0x4025c3,%esi	#将$0x4025c3赋给%esi
   0x0000000000401485 <+41>:    mov    $0x0,%eax	#将0赋给%eax
   0x000000000040148a <+46>:    call   0x400bf0 <__isoc99_sscanf@plt>
   0x000000000040148f <+51>:    cmp    $0x5,%eax	#比较%eax的值和0x5
   0x0000000000401492 <+54>:    jg     0x401499 <read_six_numbers+61>
   0x0000000000401494 <+56>:    call   0x40143a <explode_bomb>
   0x0000000000401499 <+61>:    add    $0x18,%rsp	#回收空间
   0x000000000040149d <+65>:    ret
End of assembler dump.
```

**__isoc99_sscanf 函数**：

- `__isoc99_sscanf` 是 `sscanf` 函数的实现，它用于从字符串中解析格式化输入。
- `sscanf` 函数的原型为：`int sscanf(const char *str, const char *format, ...);`，它从 `str` 中读取数据并根据 `format` 字符串进行解析，将结果存储在后面的参数中。
- 返回值是解析成功的项数，储存在`%rax`中，此处共6项，所以有一个解析项数不满足大于5则爆炸的指令。

经过计算，我们发现`sscanf`将解析结果**按照顺序依次指向了内存中一段连续的地址，也就是从`%rsi`开始增长的数组： `%rdx`（`%rsi` 指向的地址）、`%rcx`（`%rsi + 4`）、`%r8`（`%rsi + 8`）、`%r9`（`%rsi + 12`）、`%rsp`（`%rsi + 16`）、`%rsp + 8`（`%rsi + 20`）指向的内存位置。**而实际上，`read_six_numbers`函数执行完成回收内存空间后，受到`%rsi=%rsp`的限制，数组实际上是从`%rsp`开始。

知道这一点后，我们回过头看phase_2函数：第一个判断限制了第一个参数也就是%rsp必须为1.

画图分析，实际上原始代码可以简化为一个for循环：

![](/Pictures/CMU15-213/Bomb.jpg)

```c
int phase2(int *rsp)
{
    rsp[0] = 1;
    int rbp=7;
    for(int rbx=1;rbx < rbp; rbx++)
    {
        if(rsp[rbx] != 2*rsp[rbx-1])
        {
            explode_bomb();
        }
    }
}
```

于是我们得到了phase_2的密码：1 2 4 8 16 32.

### Phase_3

依然是先打断点和检验答案，发现无误。提取phase_3的代码：

```assembly
Dump of assembler code for function phase_3:
=> 0x0000000000400f43 <+0>:     sub    $0x18,%rsp	#申请空间
   0x0000000000400f47 <+4>:     lea    0xc(%rsp),%rcx	#%rcx=%rsp+0xc
   0x0000000000400f4c <+9>:     lea    0x8(%rsp),%rdx	#%rdx=%rsp+0x8
   0x0000000000400f51 <+14>:    mov    $0x4025cf,%esi	#%esi=$ox4025cf
   0x0000000000400f56 <+19>:    mov    $0x0,%eax	#%eax=0x0
   0x0000000000400f5b <+24>:    call   0x400bf0 <__isoc99_sscanf@plt>
   0x0000000000400f60 <+29>:    cmp    $0x1,%eax#比较%eax和0x1,%eax为解析结果
   0x0000000000400f63 <+32>:    jg     0x400f6a <phase_3+39>	#%eax大于1跳转
   0x0000000000400f65 <+34>:    call   0x40143a <explode_bomb>	#%eax小于1引爆
   0x0000000000400f6a <+39>:    cmpl   $0x7,0x8(%rsp)	#比较*(%rsp+8)与7
   0x0000000000400f6f <+44>:    ja     0x400fad <phase_3+106>	#大于7跳转，引爆
   0x0000000000400f71 <+46>:    mov    0x8(%rsp),%eax	#小于7%eax=*(%rsp+8)
   0x0000000000400f75 <+50>:    jmp    *0x402470(,%rax,8)#跳*(0x402470+8%rax)
   0x0000000000400f7c <+57>:    mov    $0xcf,%eax	#%eax=0xcf
   0x0000000000400f81 <+62>:    jmp    0x400fbe <phase_3+123>
   0x0000000000400f83 <+64>:    mov    $0x2c3,%eax	#%eax=0x2c3
   0x0000000000400f88 <+69>:    jmp    0x400fbe <phase_3+123>
   0x0000000000400f8a <+71>:    mov    $0x100,%eax	#%eax=0x100
   0x0000000000400f8f <+76>:    jmp    0x400fbe <phase_3+123>
   0x0000000000400f91 <+78>:    mov    $0x185,%eax	#%eax=0x185
   0x0000000000400f96 <+83>:    jmp    0x400fbe <phase_3+123>
   0x0000000000400f98 <+85>:    mov    $0xce,%eax	#%eax=0xce
   0x0000000000400f9d <+90>:    jmp    0x400fbe <phase_3+123>
   0x0000000000400f9f <+92>:    mov    $0x2aa,%eax	#%eax=0x2aa
   0x0000000000400fa4 <+97>:    jmp    0x400fbe <phase_3+123>
   0x0000000000400fa6 <+99>:    mov    $0x147,%eax	#%eax=0x147
   0x0000000000400fab <+104>:   jmp    0x400fbe <phase_3+123>
   0x0000000000400fad <+106>:   call   0x40143a <explode_bomb>
   0x0000000000400fb2 <+111>:   mov    $0x0,%eax	#%eax=0x0
   0x0000000000400fb7 <+116>:   jmp    0x400fbe <phase_3+123>
   0x0000000000400fb9 <+118>:   mov    $0x137,%eax	#%eax=0x137
   0x0000000000400fbe <+123>:   cmp    0xc(%rsp),%eax	#比较%eax和*(rsp+0xc)
   0x0000000000400fc2 <+127>:   je     0x400fc9 <phase_3+134>	#相等跳转结束
   0x0000000000400fc4 <+129>:   call   0x40143a <explode_bomb>	#不等爆炸
   0x0000000000400fc9 <+134>:   add    $0x18,%rsp
   0x0000000000400fcd <+138>:   ret
End of assembler dump.
```

这里要注意，一次是无法显示全部代码的。我们需要执行一次RET直到显示`End of assembler dump.`才算完成。

还是继续分析这段代码的逻辑，前段调用`sscanf`，而判断eax是否大于1说明`sccanf`解析出两个结果，存放在`%rdx(%rsp+8)`和`%rcx(%rsp+12)`中。而format存放在地址$0x4025cf处，我们解读一下：

```
(gdb) print (char*) 0x4025cf
$1 = 0x4025cf "%d %d"
```

发现是两个整数，也就是说第三关的密码是两个整数，解析完成后`%eax=2`。

然后我们观察+39行，比较大于7会引爆，所以我们的第一个数一定不大于7.之后将第一个数赋值给`%eax`，并跳转到`*(0x402470+8%rax)`处，这里由于置零，`%rax=%eax`。

跳转之后下面的代码都是一行赋值一行跳转。我们聚焦于最后结束的条件：`比较%eax和*(rsp+0xc)`，也就是说我们跳转的`*(0x402470+8%rax)`会告诉我们跳转到其中一行，给`%eax`赋值后使其于第二个数相等。也就是说，第二个数就是上面`%eax`赋值的数的其中之一。

其实做到这一步，由于我们知道第一个数不大于7，第二个数在上面的赋值中，已经可以穷举了🤣。但是我们还是先去找到`*(0x402470+8%rax)`的地址。

我们把第一个数从0到7依次赋值，好吧，其实这道题多解。解析出来发现其实每一个值正好对应了一个赋值语句。

```assembly
(gdb) print /x *0x402470
$1 = 0x400f7c
```

而`0x400f7c`对应的语句：

```assembly
0x0000000000400f7c <+57>:    mov    $0xcf,%eax	#%eax=0xcf
0x0000000000400f81 <+62>:    jmp    0x400fbe <phase_3+123>
```

故这一组答案为`0 207`

给出所有八组答案：

0 207；1 311；2 707；3 256；4 389；5 206；6 682；7 327。

### Phase_4

打断点解析：

```assembly
Dump of assembler code for function phase_4:
=> 0x000000000040100c <+0>:     sub    $0x18,%rsp
   0x0000000000401010 <+4>:     lea    0xc(%rsp),%rcx
   0x0000000000401015 <+9>:     lea    0x8(%rsp),%rdx
   0x000000000040101a <+14>:    mov    $0x4025cf,%esi
   0x000000000040101f <+19>:    mov    $0x0,%eax
   0x0000000000401024 <+24>:    call   0x400bf0 <__isoc99_sscanf@plt>
   0x0000000000401029 <+29>:    cmp    $0x2,%eax	#比较%eax和2
   0x000000000040102c <+32>:    jne    0x401035 <phase_4+41>	#不相等跳转，爆
   0x000000000040102e <+34>:    cmpl   $0xe,0x8(%rsp)	#比较第一个数和14
   0x0000000000401033 <+39>:    jbe    0x40103a <phase_4+46>#小于等于跳转
   0x0000000000401035 <+41>:    call   0x40143a <explode_bomb>
   0x000000000040103a <+46>:    mov    $0xe,%edx	#%edx=14
   0x000000000040103f <+51>:    mov    $0x0,%esi	#%esi=0
   0x0000000000401044 <+56>:    mov    0x8(%rsp),%edi	#%edi=第一个数
   0x0000000000401048 <+60>:    call   0x400fce <func4>
   0x000000000040104d <+65>:    test   %eax,%eax	#检测正负
   0x000000000040104f <+67>:    jne    0x401058 <phase_4+76>	#非零爆炸
   0x0000000000401051 <+69>:    cmpl   $0x0,0xc(%rsp)	#比较第二个数和0
   0x0000000000401056 <+74>:    je     0x40105d <phase_4+81>	#相等结束
   0x0000000000401058 <+76>:    call   0x40143a <explode_bomb>
   0x000000000040105d <+81>:    add    $0x18,%rsp
   0x0000000000401061 <+85>:    ret
End of assembler dump.
```

很像啊，和phase_3，连format都是用的同一个。所以这次的密码也是两个整数。由已知信息，第一个数小于等于14，而第一、二个数要为0（经过func4后）.

这里涉及到一个关键中间函数`func4`，我们调出来看一下：

```assembly
Dump of assembler code for function func4:
=> 0x0000000000400fce <+0>:     sub    $0x8,%rsp
   0x0000000000400fd2 <+4>:     mov    %edx,%eax	#%eax=%edx=14
   0x0000000000400fd4 <+6>:     sub    %esi,%eax	#%eax -= %esi，而%esi=0
   0x0000000000400fd6 <+8>:     mov    %eax,%ecx	#%ecx=%eax=14
   0x0000000000400fd8 <+10>:    shr    $0x1f,%ecx	#逻辑右移%ecx中31位
   0x0000000000400fdb <+13>:    add    %ecx,%eax	#%eax += %ecx
   0x0000000000400fdd <+15>:    sar    %eax			#%eax=%eax/2
   0x0000000000400fdf <+17>:    lea    (%rax,%rsi,1),%ecx	#%ecx=%rax+%rsi
   0x0000000000400fe2 <+20>:    cmp    %edi,%ecx	#比较%ecx和%edi
   0x0000000000400fe4 <+22>:    jle    0x400ff2 <func4+36>#小于等于跳转
   0x0000000000400fe6 <+24>:    lea    -0x1(%rcx),%edx#大于%edx=%rcx-1
   0x0000000000400fe9 <+27>:    call   0x400fce <func4>#递归调用
   0x0000000000400fee <+32>:    add    %eax,%eax	#%eax*=2
   0x0000000000400ff0 <+34>:    jmp    0x401007 <func4+57>	#跳转，结束
   0x0000000000400ff2 <+36>:    mov    $0x0,%eax	#%eax=0
   0x0000000000400ff7 <+41>:    cmp    %edi,%ecx	#比较%ecx与%edi
   0x0000000000400ff9 <+43>:    jge    0x401007 <func4+57>	#大于等于，结束
   0x0000000000400ffb <+45>:    lea    0x1(%rcx),%esi	#否则%esi=%rcx+1
   0x0000000000400ffe <+48>:    call   0x400fce <func4>	#递归调用
   0x0000000000401003 <+53>:    lea    0x1(%rax,%rax,1),%eax#%eax=2%rax+1
   0x0000000000401007 <+57>:    add    $0x8,%rsp
   0x000000000040100b <+61>:    ret
End of assembler dump.
```

这里`func4`中没有涉及到`%rsp+0xc`也就是第二个数的位置，那么我们便能确定第二个数就是0.对于第一个数，需要确定什么样的输入才能使`func4`返回0.这段函数由于涉及两部分的递归调用，较为复杂。笔者初始假设第一个数为0发现能够得到返回值为0.也就是说0 0为一组较明显的答案。实际上有不大于14的限制，逐个代入也能解出全解。但这里尝试转换为C代码。

```c
int func4()
{
    static int edx=14, esi=0, edi=x, eax, ecx;  //这里没涉及指针操作，因此设为整型而不是指针。
    eax = edx;
    eax = eax-esi;
    ecx = eax;
    ecx = sign(eax);  //ecx储存eax的正负号，1为负0为正
    eax += ecx;
    eax /= 2;
    ecx = eax+esi;
    if(ecx <= edi)
    {
        eax = 0;
        if(ecx >= edi)
        {
            return eax;
        }
        else
        {
            esi = ecx+1;
            eax = func4();
            eax = 2*eax + 1;
            return eax;
        }
    }
    else 
    {
        edx = ecx-1;
        eax = func4();
        eax = eax*2;
        return eax;
    }
}
```

继续简化,这里由于`eax`的初值是2，由我们输入决定的变量是`edi`，由观察返回值可知，与`eax`的初值无关且`eax`恒非负。因此所有由移位产生出来的判断符号的`ecx`都为0。

```c
// m代edx，n代esi
int func4(int x,int m=14,int n=0)
{
    int val; //代ecx
    val = (m+n)/2;
    if(val == x) 
        return 0;
    if(val < x)
    {
        return 2*func4(x,m,val+1)+1;
    }
    if(val > x)
    {
        return 2*func4(x,val-1,n);
    }
}
```

这样我们就能快乐的在外面套一个for循环和一个if==0来判断答案啦😎。

最终结果：

0 0；1 0；3 0；7 0

### Phase_5

解析代码：

```assembly
Dump of assembler code for function phase_5:
=> 0x0000000000401062 <+0>:     push   %rbx
   0x0000000000401063 <+1>:     sub    $0x20,%rsp
   0x0000000000401067 <+5>:     mov    %rdi,%rbx	#%rbx=%rdi
   0x000000000040106a <+8>:     mov    %fs:0x28,%rax#这里实际上是设置了一个Canary
   0x0000000000401073 <+17>:    mov    %rax,0x18(%rsp)	#*(%rsp+0x18)=%rax
   0x0000000000401078 <+22>:    xor    %eax,%eax	#%eax=0
   0x000000000040107a <+24>:    call   0x40131b <string_length>
   0x000000000040107f <+29>:    cmp    $0x6,%eax	#要求一个长度为6的字符串
   0x0000000000401082 <+32>:    je     0x4010d2 <phase_5+112>
   0x0000000000401084 <+34>:    call   0x40143a <explode_bomb>
   0x0000000000401089 <+39>:    jmp    0x4010d2 <phase_5+112>
   0x000000000040108b <+41>:    movzbl (%rbx,%rax,1),%ecx	#%ecx=%rbx+%rax，保留低8位
   0x000000000040108f <+45>:    mov    %cl,(%rsp)	#*（%rsp）=%cl
   0x0000000000401092 <+48>:    mov    (%rsp),%rdx	#%rdx=*（%rsp）
   0x0000000000401096 <+52>:    and    $0xf,%edx	#获取当前字符后四位，设为0xk
   0x0000000000401099 <+55>:    movzbl 0x4024b0(%rdx),%edx#%edx=*(%rdx+..)保留低8位，保留后%edx=0x4024bk
   0x00000000004010a0 <+62>:    mov    %dl,0x10(%rsp,%rax,1)#*(%rsp+%rax+0x10)=%dl，%dl也就是%rdx低8位0xk
   0x00000000004010a4 <+66>:    add    $0x1,%rax	#%rax+=1
   0x00000000004010a8 <+70>:    cmp    $0x6,%rax	#比较%rax与6
   0x00000000004010ac <+74>:    jne    0x40108b <phase_5+41>	#不等，跳回+41
   0x00000000004010ae <+76>:    movb   $0x0,0x16(%rsp)	#相等，*(%rsp+0x16)=0
   0x00000000004010b3 <+81>:    mov    $0x40245e,%esi	#%esi=0x40245e
   0x00000000004010b8 <+86>:    lea    0x10(%rsp),%rdi	#%rdi=%rsp+0x10
   0x00000000004010bd <+91>:    call   0x401338 <strings_not_equal>
   0x00000000004010c2 <+96>:    test   %eax,%eax	#判断字符是否相等，相等即结束。
   0x00000000004010c4 <+98>:    je     0x4010d9 <phase_5+119>
   0x00000000004010c6 <+100>:   call   0x40143a <explode_bomb>
   0x00000000004010cb <+105>:   nopl   0x0(%rax,%rax,1)
   0x00000000004010d0 <+110>:   jmp    0x4010d9 <phase_5+119>
   0x00000000004010d2 <+112>:   mov    $0x0,%eax	#%eax=0
   0x00000000004010d7 <+117>:   jmp    0x40108b <phase_5+41>
   0x00000000004010d9 <+119>:   mov    0x18(%rsp),%rax
   0x00000000004010de <+124>:   xor    %fs:0x28,%rax
   0x00000000004010e7 <+133>:   je     0x4010ee <phase_5+140>
   0x00000000004010e9 <+135>:   call   0x400b30 <__stack_chk_fail@plt>
   0x00000000004010ee <+140>:   add    $0x20,%rsp
   0x00000000004010f2 <+144>:   pop    %rbx
   0x00000000004010f3 <+145>:   ret
End of assembler dump.
```

同时解析相关的函数`string_length`：

```assembly
Dump of assembler code for function string_length:
=> 0x000000000040131b <+0>:     cmpb   $0x0,(%rdi)	#比较*(%rdi)与0
   0x000000000040131e <+3>:     je     0x401332 <string_length+23>	#相等，结束
   0x0000000000401320 <+5>:     mov    %rdi,%rdx	#不等，%rdx=%rdi
   0x0000000000401323 <+8>:     add    $0x1,%rdx	#%rdx += 1
   0x0000000000401327 <+12>:    mov    %edx,%eax	#%eax=%edx
   0x0000000000401329 <+14>:    sub    %edi,%eax	#%eax -= %edi
   0x000000000040132b <+16>:    cmpb   $0x0,(%rdx)	#比较*（%rdx）和0
   0x000000000040132e <+19>:    jne    0x401323 <string_length+8>	#不等，跳回
   0x0000000000401330 <+21>:    repz ret	#相等，返回
   0x0000000000401332 <+23>:    mov    $0x0,%eax
   0x0000000000401337 <+28>:    ret
End of assembler dump.
```

结合两段代码，我们得到`string_length`函数的逻辑：从我们的输入`%rdi`中读取数据，返回字符串的长度（这里通过连续+1判断是否为零也就是为空来判断是否储存字符，其中的`%eax=%edx-%edi`就是字符串长度）。判断`%eax`是否为6说明答案是一个长度为6的字符串。

下面将`%eax`置零后跳转到+41继续。之后经过几步将我们输入的字符串存到`%rdx`中，并只改变了低4位。每次利用`%rax+1`将储存在`0x4024bk`的字符截取得到比对字符串，转到`%rsp+0x10-%rsp+0x15`处。重新将字符串的首地址`%rsp+0x10`赋给`%rdi`作为`strings_not_equal`函数的第一个参数，调用该函数。

```assembly
Dump of assembler code for function strings_not_equal:
=> 0x0000000000401338 <+0>:     push   %r12
   0x000000000040133a <+2>:     push   %rbp
   0x000000000040133b <+3>:     push   %rbx
   0x000000000040133c <+4>:     mov    %rdi,%rbx	#%rbx储存第一个参数
   0x000000000040133f <+7>:     mov    %rsi,%rbp
   0x0000000000401342 <+10>:    call   0x40131b <string_length>
   0x0000000000401347 <+15>:    mov    %eax,%r12d	#%r12d=%eax=6
   0x000000000040134a <+18>:    mov    %rbp,%rdi	#%rbp为第二个参数赋给%rdi
   0x000000000040134d <+21>:    call   0x40131b <string_length>
   0x0000000000401352 <+26>:    mov    $0x1,%edx	#%edx=1
   0x0000000000401357 <+31>:    cmp    %eax,%r12d	#此处%eax为第二个参数的长度，这几步是判断两者是否等长。不等长跳转返回1.
   0x000000000040135a <+34>:    jne    0x40139b <strings_not_equal+99>
   0x000000000040135c <+36>:    movzbl (%rbx),%eax
   0x000000000040135f <+39>:    test   %al,%al
   0x0000000000401361 <+41>:    je     0x401388 <strings_not_equal+80>
   0x0000000000401363 <+43>:    cmp    0x0(%rbp),%al
   0x0000000000401366 <+46>:    je     0x401372 <strings_not_equal+58>
   0x0000000000401368 <+48>:    jmp    0x40138f <strings_not_equal+87>
   0x000000000040136a <+50>:    cmp    0x0(%rbp),%al
   0x000000000040136d <+53>:    nopl   (%rax)
   0x0000000000401370 <+56>:    jne    0x401396 <strings_not_equal+94>
   0x0000000000401372 <+58>:    add    $0x1,%rbx
   0x0000000000401376 <+62>:    add    $0x1,%rbp
   0x000000000040137a <+66>:    movzbl (%rbx),%eax
   0x000000000040137d <+69>:    test   %al,%al
   0x000000000040137f <+71>:    jne    0x40136a <strings_not_equal+50>
   0x0000000000401381 <+73>:    mov    $0x0,%edx
   0x0000000000401386 <+78>:    jmp    0x40139b <strings_not_equal+99>
   0x0000000000401388 <+80>:    mov    $0x0,%edx
   0x000000000040138d <+85>:    jmp    0x40139b <strings_not_equal+99>
   0x000000000040138f <+87>:    mov    $0x1,%edx
   0x0000000000401394 <+92>:    jmp    0x40139b <strings_not_equal+99>
   0x0000000000401396 <+94>:    mov    $0x1,%edx
   0x000000000040139b <+99>:    mov    %edx,%eax
   0x000000000040139d <+101>:   pop    %rbx
   0x000000000040139e <+102>:   pop    %rbp
   0x000000000040139f <+103>:   pop    %r12
   0x00000000004013a1 <+105>:   ret
End of assembler dump.
```

整个函数先比较长度，后通过字符一一使用`test`比较是否相等。总体而言不等返回1，相等返回0.

比较所用的函数储存在`%esi`中，而利用地址0x40245e，我们解析出用于比较的字符：

```assembly
(gdb) print (char*) 0x40245e
$1 = 0x40245e "flyers"
```

同时我们解析位于`0x4024b0`处的字符，也就是我们需要截取出flyers的源字符：

```assembly
(gdb) print (char*) 0x4024b0
$2 = 0x4024b0 <array> "maduiersnfotvbylSo you think you can stop the bomb with ctrl-c, do you?"
```

观察可得，要凑齐flyers，需要源字符串的index为9、15、14、5、6、7。而我们输入的六个字符需要满足ASCII码的低4位要译出对应数字作为偏移量。查表可得答案为：

**ionefg.**

### Phase_6

最后一关无疑是最复杂的一关，源码的量都是巨大的：

很长，但其中的逻辑并不困难，我们只需要~~亿点~~耐心慢慢看就行（）。

```assembly
Dump of assembler code for function phase_6:
=> 0x00000000004010f4 <+0>:     push   %r14
   0x00000000004010f6 <+2>:     push   %r13
   0x00000000004010f8 <+4>:     push   %r12
   0x00000000004010fa <+6>:     push   %rbp
   0x00000000004010fb <+7>:     push   %rbx
   0x00000000004010fc <+8>:     sub    $0x50,%rsp
   0x0000000000401100 <+12>:    mov    %rsp,%r13
   0x0000000000401103 <+15>:    mov    %rsp,%rsi
   0x0000000000401106 <+18>:    call   0x40145c <read_six_numbers>
   0x000000000040110b <+23>:    mov    %rsp,%r14	#%r14=%rsp
   0x000000000040110e <+26>:    mov    $0x0,%r12d	#r12d=0
   0x0000000000401114 <+32>:    mov    %r13,%rbp	#%rbp=%r13
   0x0000000000401117 <+35>:    mov    0x0(%r13),%eax	#%eax=*(%r13)，第一个数
   0x000000000040111b <+39>:    sub    $0x1,%eax	#%eax -= 1
   0x000000000040111e <+42>:    cmp    $0x5,%eax	#比较%eax和5
   0x0000000000401121 <+45>:    jbe    0x401128 <phase_6+52>	#小于等于，继续
   0x0000000000401123 <+47>:    call   0x40143a <explode_bomb>
   0x0000000000401128 <+52>:    add    $0x1,%r12d	#%r12d += 1
   0x000000000040112c <+56>:    cmp    $0x6,%r12d	#比较%r12d与6
   0x0000000000401130 <+60>:    je     0x401153 <phase_6+95>	#相等跳转
   0x0000000000401132 <+62>:    mov    %r12d,%ebx	#不等，%ebx=%r12d=1
   0x0000000000401135 <+65>:    movslq %ebx,%rax	#%rax=%ebx=%r12d
   0x0000000000401138 <+68>:    mov    (%rsp,%rax,4),%eax#%eax=%rsp+4%rax，相当于移动到下一个数
   0x000000000040113b <+71>:    cmp    %eax,0x0(%rbp)#比较*(%rbp)与%eax
   0x000000000040113e <+74>:    jne    0x401145 <phase_6+81>#不等，继续运行
   0x0000000000401140 <+76>:    call   0x40143a <explode_bomb>
   0x0000000000401145 <+81>:    add    $0x1,%ebx	#%ebx +=1
   0x0000000000401148 <+84>:    cmp    $0x5,%ebx	#比较%ebx和5
   0x000000000040114b <+87>:    jle    0x401135 <phase_6+65>#小于等于，跳回
   0x000000000040114d <+89>:    add    $0x4,%r13	#大于5，%r13=4
   0x0000000000401151 <+93>:    jmp    0x401114 <phase_6+32>
   0x0000000000401153 <+95>:    lea    0x18(%rsp),%rsi
   0x0000000000401158 <+100>:   mov    %r14,%rax
   0x000000000040115b <+103>:   mov    $0x7,%ecx
   0x0000000000401160 <+108>:   mov    %ecx,%edx
   0x0000000000401162 <+110>:   sub    (%rax),%edx	#7-%rax，将结果放回
   0x0000000000401164 <+112>:   mov    %edx,(%rax)
   0x0000000000401166 <+114>:   add    $0x4,%rax	#移动到下一个数
   0x000000000040116a <+118>:   cmp    %rsi,%rax
   0x000000000040116d <+121>:   jne    0x401160 <phase_6+108>
   0x000000000040116f <+123>:   mov    $0x0,%esi
   0x0000000000401174 <+128>:   jmp    0x401197 <phase_6+163>
   0x0000000000401176 <+130>:   mov    0x8(%rdx),%rdx	#移动到下一个节点
   0x000000000040117a <+134>:   add    $0x1,%eax
   0x000000000040117d <+137>:   cmp    %ecx,%eax
   0x000000000040117f <+139>:   jne    0x401176 <phase_6+130>
   0x0000000000401181 <+141>:   jmp    0x401188 <phase_6+148>
   0x0000000000401183 <+143>:   mov    $0x6032d0,%edx
   0x0000000000401188 <+148>:   mov    %rdx,0x20(%rsp,%rsi,2)#赋值为输入序号
   0x000000000040118d <+153>:   add    $0x4,%rsi
   0x0000000000401191 <+157>:   cmp    $0x18,%rsi
   0x0000000000401195 <+161>:   je     0x4011ab <phase_6+183>
   0x0000000000401197 <+163>:   mov    (%rsp,%rsi,1),%ecx
   0x000000000040119a <+166>:   cmp    $0x1,%ecx
   0x000000000040119d <+169>:   jle    0x401183 <phase_6+143>
   0x000000000040119f <+171>:   mov    $0x1,%eax
   0x00000000004011a4 <+176>:   mov    $0x6032d0,%edx
   0x00000000004011a9 <+181>:   jmp    0x401176 <phase_6+130>
   0x00000000004011ab <+183>:   mov    0x20(%rsp),%rbx#依次判别小于
   0x00000000004011b0 <+188>:   lea    0x28(%rsp),%rax
   0x00000000004011b5 <+193>:   lea    0x50(%rsp),%rsi
   0x00000000004011ba <+198>:   mov    %rbx,%rcx
   0x00000000004011bd <+201>:   mov    (%rax),%rdx
   0x00000000004011c0 <+204>:   mov    %rdx,0x8(%rcx)
   0x00000000004011c4 <+208>:   add    $0x8,%rax
   0x00000000004011c8 <+212>:   cmp    %rsi,%rax
   0x00000000004011cb <+215>:   je     0x4011d2 <phase_6+222>
   0x00000000004011cd <+217>:   mov    %rdx,%rcx
   0x00000000004011d0 <+220>:   jmp    0x4011bd <phase_6+201>
   0x00000000004011d2 <+222>:   movq   $0x0,0x8(%rdx)
   0x00000000004011da <+230>:   mov    $0x5,%ebp
   0x00000000004011df <+235>:   mov    0x8(%rbx),%rax
   0x00000000004011e3 <+239>:   mov    (%rax),%eax
   0x00000000004011e5 <+241>:   cmp    %eax,(%rbx)
   0x00000000004011e7 <+243>:   jge    0x4011ee <phase_6+250>
   0x00000000004011e9 <+245>:   call   0x40143a <explode_bomb>
   0x00000000004011ee <+250>:   mov    0x8(%rbx),%rbx
   0x00000000004011f2 <+254>:   sub    $0x1,%ebp
   0x00000000004011f5 <+257>:   jne    0x4011df <phase_6+235>
   0x00000000004011f7 <+259>:   add    $0x50,%rsp
   0x00000000004011fb <+263>:   pop    %rbx
   0x00000000004011fc <+264>:   pop    %rbp
   0x00000000004011fd <+265>:   pop    %r12
   0x00000000004011ff <+267>:   pop    %r13
   0x0000000000401201 <+269>:   pop    %r14
   0x0000000000401203 <+271>:   ret
End of assembler dump.
```

分阶段分析：

- 从开头到+18，出现函数`read_six_numbers`，说明我们的输入依旧是6个数字，而6个数字储存在`%rsp`开头的数组中。

- +39行`%eax-1`，控制每个数不大于6。

- +60 ~ +95，这里主要用于判断两数之间是否相等。

- +108 ~ +128，将六个数都转化为7-%rax

- +130 ~ +181，我们发现一个地址：0x6032d0，打印出来

  ```assembly
  (gdb) x/24w 0x6032d0
  0x6032d0 <node1>:       332     1       6304480 0
  0x6032e0 <node2>:       168     2       6304496 0
  0x6032f0 <node3>:       924     3       6304512 0
  0x603300 <node4>:       691     4       6304528 0
  0x603310 <node5>:       477     5       6304544 0
  0x603320 <node6>:       443     6       0       0
  ```

  发现了六个节点，正说明这个地址储存的是一个链表。而`mov    0x8(%rdx),%rdx`代表移动到下一个节点。其中，序号1~6是这些节点的编号，前面的数字是储存的值。

  得知这一点后，我们分析代码，得知这一段代码的核心是利用我们输入的编号序列对其进行重新排序（也就是%rsp，%rsi和%rdx之间的互相操作）。

- +183后：判断链表是否满足降序。因此在前面我们需要满足降序，而降序的链表存放的数据是经过7-x处理的。因此按源值降序为：3，4，5，6，1，2，经过处理后的降序为4，3，2，1，6，5。这就是我们的结果。

最后检验结果：

```assembly
(gdb) run ans.txt
Starting program: /home/usr/CMU15-213 labs/bomb/bomb ans.txt
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Welcome to my fiendish little bomb. You have 6 phases with
which to blow yourself up. Have a nice day!
Phase 1 defused. How about the next one?
That's number 2.  Keep going!
Halfway there!
So you got that one.  Try this one.
Good work!  On to the next...
Congratulations! You've defused the bomb!
```

完成！