# Git学习笔记
---
## 1. Git简介
Git是目前世界上最先进的分布式版本控制系统，与此相对的是集中式的版本控制系统CVS和SVN。  
集中式版本控制系统的版本库放置在中央服务器，用户使用客户机从服务器获得最新版本，在编辑以后再将文件推送回中央服务器。  
分布式版本控制系统的版本库存储在每个客户机上，互相推送修改后的版本来获取对方的修改。 
### 1.1 Git的安装  
#### 1.1.1 在Linux上安装Git
在Debian或Ubuntu Linux，通过一条`sudo apt-get install git`就可以直接完成Git的安装。  
#### 1.1.2在Windows上安装Git  
安装完成后，在开始菜单里找到“Git”->“Git Bash”，蹦出一个类似命令行窗口的东西
![alt GitBash](GitNoteImage.png)
安装完成后，还需要输入user name和Email地址，在命令行输入  
```
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```
注意`git config`命令的`--global`参数，用了这个参数，表示你这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址。
### 1.2 创建版本库
- **注意**，创建仓库前需要确认Git Bash程序当前工作的目录为需要创建仓库的路径，否则需要`cd`到该路径.
- 通过`git init`命令把这个目录变成Git可以管理的仓库
```
$ git init
Initialized empty Git repository in D:/CodeFile/learngit/.git/
```
`.git`目录一般是隐藏的，用于跟踪管理版本，一般不要修改
***
## 2. Git的简单语法及功能简介
### 2.1 把文件添加到仓库  
**主要涉及的命令`add`, `commit`**  
- 首先，我们需要把文件放在仓库目录下（或子目录），以文件`readme.txt`为例,其内容如下：  
```
Git is a distributed version control system.
Git is free software
```
- 然后使用`git add`命令把文件添加到仓库，
```
$ git add readme.txt
```
执行上面的命令，没有任何显示，这表示一切正常，因为Unixd的哲学是——“没有消息就是好消息”。  
- 用命令`git commit`告诉Git，把文件提交到仓库：
```
$ git commit -m "wrote a readme file"
[master (root-commit) 6a1f78b] wrote a readme file
1 file changed, 2 insertions(+)
create mode 100644 readme.txt
```
简单解释一下`git commit`命令，`-m`后面输入的是本次提交的说明，可以输入任意内容，当然最好是有意义的，这样你就能从历史记录里方便地找到改动记录。
`git commit`命令执行成功后会告诉你，`1 file changed`：1个文件被改动（我们新添加的`readme.txt`文件）；`2 insertions`：插入了两行内容（`readme.txt`有两行内容）。  
为什么Git添加文件需要`add`，`commit`一共两步呢？因为`commit`可以一次提交很多文件，所以你可以多次`add`不同的文件，比如：  
```
$ git add file1.txt
$ git add file2.txt file3.txt
$ git commit -m "add 3 files."
```
### 2.2 使用Git进行版本控制
- 使用`status`查看仓库状态
  在`readme.txt`经过修改后，我们运行`git status`命令，可以得到以下结果（仅截取部分输出）：
  ```
  $ git status
  On branch master
  Changes not staged for commit:
  ...
  ```
上面的输出告诉我们，`readme.txt`被修改了，但是还没有加载到暂存区准备提交。  
- 如果`git status`告诉你文件被修改了，用`git diff`可以查看修改的内容。
#### 2.2.1 版本回退
- 使用`git log`查看我们查看历史版本
```
$ git log
commit b34ed6124918e2cf82f4b0121ac183b05b4ac2b5 (HEAD -> master)
Author: PQball <pingjin291@gmail.com>
Date:   Sun Mar 6 17:16:30 2022 +0800

    append GPL

commit 16c1af4a6119195180d936c158a50b94792b6cdf
Author: PQball <pingjin291@gmail.com>
Date:   Sun Mar 6 17:11:29 2022 +0800

    add distributed

commit 6a1f78b8613ef825ddc889b23e6c386ccaaa6ba8
Author: PQball <pingjin291@gmail.com>
Date:   Sun Mar 6 17:02:55 2022 +0800

    wrote a readme file
```
`git log`命令显示最近到最远的提交日志，我们可以看到3次提交，最近的一次是`append GPL`，上一次是`add distributed`，最早的一次是`wrote a readme file`。  
如果嫌输出信息太多，看得眼花缭乱的，可以试试加上`--pretty=oneline`参数:
```
$ git log --pretty=oneline
```
其中的`commit b34ed6...`是`commit id`(版本号)，是一个SHA计算出来的十六进制数字。
- 使用`git reset`进行版本回退  
在Git中，用`HEAD`表示当前版本，上一个版本表示为`HEAD^`,上上个版本表示为`HEAD^^`，往前100个版本表示为`HEAD~100`。
```
$ git reset --hard HEAD^
```
当然，也可以使用`commit id`的方式，版本号不用写全，前几位就可以：
```
$ git reset --hard b34ed612
```
- 使用`git reflog`查看历史`commit id`
```
$ git reflog
```
#### 2.2.2 工作区和暂存区
- **工作区（Working Directory）：**
  就是你在电脑里能看到的目录，比如我的learngit文件夹就是一个工作区：
- **版本库：**
  工作区有一个隐藏目录.git，这个不算工作区，而是Git的版本库。  
  Git的版本库里存了很多东西，其中最重要的就是称为stage（或者叫index）的暂存区，还有Git为我们自动创建的第一个分支master，以及指向master的一个指针叫HEAD。  
  前面讲了我们把文件往Git版本库里添加的时候，是分两步执行的：  
    >第一步是用git add把文件添加进去，实际上就是把文件修改添加到暂存区；   
    >第二步是用git commit提交更改，实际上就是把暂存区的所有内容提交到当前分支。

    因为我们创建Git版本库时，Git自动为我们创建了唯一一个master分支，所以，现在，git commit就是往master分支上提交更改。
#### 2.2.3 管理修改
- Git管理的是修改而不是文件
- 使用`git diff HEAD -- readme.txt`命令可以查看工作区和版本库里面最新版本的区别：
```
  $ git diff HEAD -- readme.txt
  diff --git a/readme.txt b/readme.txt
  index db28b2c..9a8b341 100644
  --- a/readme.txt
  +++ b/readme.txt
  @@ -1,4 +1,4 @@
   Git is a distributed version control system.Git is free software distributed under the GPL.
   Git has a mutable index called stage.
   Git tracks changes.
   \ No newline at end of file
   +Git tracks changes of files.
   \ No newline at end of file
``` 
#### 2.2.4 撤销修改
- `git checkout -- file`可以丢弃工作区的修改:
```
$ git checkout -- readme.txt
```
命令`git checkout -- readme.txt`意思就是，把`readme.txt`文件在工作区的修改全部撤销，这里有两种情况：  
>一种是`readme.txt`自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态;  
一种是`readme.txt`已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态。  

总之，就是让这个文件回到最近一次`git commit`或`git add`时的状态。
- `git reset HEAD <file>`可以把暂存区的修改撤销掉（unstage），重新放回工作区：
```
$ git reset HEAD readme.txt
```
- **小结**：
>场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令git checkout -- file。  
场景2：当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，第一步用命令git reset HEAD <file>，就回到了场景1，第二步按场景1操作。  
场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，参考版本回退一节，不过前提是没有推送到远程库。