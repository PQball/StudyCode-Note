# MySQL笔记
## 1.MySQL管理
### 1.1 登录MySQL
当 MySQL 服务已经运行时, 我们可以通过 MySQL 自带的客户端工具登录到 MySQL 数据库中, 首先打开命令提示符, 输入以下格式的命名:

```mysql
mysql -h 主机名 -u 用户名 -p
```
参数说明：
>`-h`: 指定客户端所要登录的MySQL主机名, 登录本机
`-h` : 指定客户端所要登录的 MySQL 主机名, 登录本机localhost 或 127.0.0.1)该参数可以省略;
`-u `: 登录的用户名;
`-p` : 告诉服务器将会使用一个密码来登录, 如果所要登录的用户名密码为空, 可以忽略此选项。

如果我们要登录本机的 MySQL 数据库，只需要输入以下命令即可：
```
mysql -u root -p
```
按回车确认, 如果安装正确且 MySQL 正在运行, 会得到以下响应:
```
Enter password:
```
若密码存在, 输入密码登录, 不存在则直接按回车登录。登录成功后你将会看到`Welcome to the MySQL monitor... `的提示语。
然后命令提示符会一直以`mysql>`加一个闪烁的光标等待命令的输入, 输入`exit`或`quit`退出登录。

### 1.2 MySQL管理
#### 启动及关闭 MySQL 服务器
在 Windows 系统下，打开命令窗口(cmd)，进入 MySQL 安装目录的 bin 目录。

启动：
```mysql
cd c:/mysql/bin
mysqld --console
```

关闭：
```
cd c:/mysql/bin
mysqladmin -uroot shutdown
```

#### MySQL 用户设置
如果你需要添加MySQL用户，你只需要在mysql数据库中的 user表添加新用户即可。
以下为添加用户的实例，用户名为guest，密码为guest123，并授权用户可进行 SELECT, INSERT 和 UPDATE操作权限：
```
root@host# mysql -u root -p
Enter password:*******
mysql> use mysql;
Database changed

mysql> INSERT INTO user 
          (host, user, password, 
           select_priv, insert_priv, update_priv) 
           VALUES ('localhost', 'guest', 
           PASSWORD('guest123'), 'Y', 'Y', 'Y');
Query OK, 1 row affected (0.20 sec)

mysql> FLUSH PRIVILEGES;
Query OK, 1 row affected (0.01 sec)

mysql> SELECT host, user, password FROM user WHERE user = 'guest';
+-----------+---------+------------------+
| host      | user    | password         |
+-----------+---------+------------------+
| localhost | guest | 6f8c114b58f2ce9e |
+-----------+---------+------------------+
1 row in set (0.00 sec)
```

在添加用户时，请注意使用MySQL提供的`PASSWORD()`函数来对密码进行加密。 你可以在以上实例看到用户密码加密后为：`6f8c114b58f2ce9e`.
__注意__：在 MySQL5.7 中 user表的`password`已换成了`authentication_string`。
__注意__：`password()`加密函数已经在 8.0.11 中移除了，可以使用`MD5()`函数代替。
__注意__：在注意需要执行`FLUSH PRIVILEGES`语句。 这个命令执行后会重新载入授权表。如果你不使用该命令，你就无法使用新创建的用户来连接mysql服务器，除非你重启mysql服务器。
你可以在创建用户时，为用户指定权限，在对应的权限列中，在插入语句中设置为 'Y' 即可，用户权限列表如下：
>`Select_priv`
`Insert_priv`
`Update_priv`
`Delete_priv`
`Create_priv`
`Drop_priv`
`Reload_priv`
`Shutdown_priv`
`Process_priv`
`File_priv`
`Grant_priv`
`References_priv`
`Index_priv`
`Alter_priv`

另外一种添加用户的方法为通过SQL的`GRANT`命令，以下命令会给指定数据库`TUTORIALS`添加用户`zara` ，密码为`zara123`。
```
root@host# mysql -u root -p
Enter password:*******
mysql> use mysql;
Database changed

mysql> GRANT SELECT,INSERT,UPDATE,DELETE,CREATE,DROP
    -> ON TUTORIALS.*
    -> TO 'zara'@'localhost'
    -> IDENTIFIED BY 'zara123';
```

#### 管理MySQL的命令  
以下列出了使用Mysql数据库过程中常用的命令：
##### USE 数据库名 :
选择要操作的Mysql数据库，使用该命令后所有Mysql命令都只针对该数据库。
```
mysql> use RUNOOB;
Database changed
```

##### USE 数据库名 :
列出 MySQL 数据库管理系统的数据库列表。
```
mysql> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| RUNOOB             |
| cdcol              |
| mysql              |
| onethink           |
| performance_schema |
| phpmyadmin         |
| test               |
| wecenter           |
| wordpress          |
+--------------------+
10 rows in set (0.02 sec)
```

##### SHOW TABLES:
显示指定数据库的所有表，使用该命令前需要使用 use 命令来选择要操作的数据库。
```
mysql> use RUNOOB;
Database changed
mysql> SHOW TABLES;
+------------------+
| Tables_in_runoob |
+------------------+
| employee_tbl     |
| runoob_tbl       |
| tcount_tbl       |
+------------------+
3 rows in set (0.00 sec)
```

##### SHOW COLUMNS FROM 数据表:
显示数据表的属性，属性类型，主键信息 ，是否为 NULL，默认值等其他信息。
```
mysql> SHOW COLUMNS FROM runoob_tbl;
+-----------------+--------------+------+-----+---------+-------+
| Field           | Type         | Null | Key | Default | Extra |
+-----------------+--------------+------+-----+---------+-------+
| runoob_id       | int(11)      | NO   | PRI | NULL    |       |
| runoob_title    | varchar(255) | YES  |     | NULL    |       |
| runoob_author   | varchar(255) | YES  |     | NULL    |       |
| submission_date | date         | YES  |     | NULL    |       |
+-----------------+--------------+------+-----+---------+-------+
4 rows in set (0.01 sec)
```

##### SHOW INDEX FROM 数据表:
显示数据表的详细索引信息，包括PRIMARY KEY（主键）。
```
mysql> SHOW INDEX FROM runoob_tbl;
+------------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+
| Table      | Non_unique | Key_name | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment |
+------------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+
| runoob_tbl |          0 | PRIMARY  |            1 | runoob_id   | A         |           2 |     NULL | NULL   |      | BTREE      |         |               |
+------------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+
1 row in set (0.00 sec)
```

##### SHOW TABLE STATUS [FROM db_name] [LIKE 'pattern'] \G:
该命令将输出Mysql数据库管理系统的性能及统计信息。
```
mysql> SHOW TABLE STATUS  FROM RUNOOB;   # 显示数据库 RUNOOB 中所有表的信息

mysql> SHOW TABLE STATUS from RUNOOB LIKE 'runoob%';     # 表名以runoob开头的表的信息
mysql> SHOW TABLE STATUS from RUNOOB LIKE 'runoob%'\G;   # 加上 \G，查询结果按列打印
```

## 2 MySQL数据库操作
### 2.1 创建数据库
我们可以在登陆 MySQL 服务后，使用 create 命令创建数据库，语法如下:
```
CREATE DATABASE 数据库名;
```

#### 使用 mysqladmin 创建数据库
使用普通用户，你可能需要特定的权限来创建或者删除 MySQL 数据库。
所以我们这边使用root用户登录，root用户拥有最高权限，可以使用 mysql `mysqladmin`命令来创建数据库。

以下命令简单的演示了创建数据库的过程，数据名为 RUNOOB:
```
[root@host]# mysqladmin -u root -p create RUNOOB
Enter password:******
```

### 2.2删除数据库
#### 2.2.1 drop命令删除数据库
drop 命令格式：
```
drop database <数据库名>;
```

#### 2.2.2 使用 mysqladmin 删除数据库
你也可以使用 mysql mysqladmin 命令在终端来执行删除命令。

以下实例删除数据库 RUNOOB(该数据库在前一章节已创建)：
```
[root@host]# mysqladmin -u root -p drop RUNOOB
Enter password:******
```
#### 2.2.3 选择数据库
在 mysql> 提示窗口中可以很简单的选择特定的数据库。你可以使用SQL命令来选择指定的数据库。
```
[root@host]# mysql -u root -p
Enter password:******
mysql> use RUNOOB;
Database changed
mysql>
```

## 3 MySQL数据类型
MySQL 支持多种类型，大致可以分为三类：数值、日期/时间和字符串(字符)类型。
### 3.1 数值类型
这些类型包括严格数值数据类型(INTEGER、SMALLINT、DECIMAL 和 NUMERIC)，以及近似数值数据类型(FLOAT、REAL 和 DOUBLE PRECISION)。

关键字INT是INTEGER的同义词，关键字DEC是DECIMAL的同义词。

BIT数据类型保存位字段值，并且支持 MyISAM、MEMORY、InnoDB 和 BDB表。

作为 SQL 标准的扩展，MySQL 也支持整数类型 TINYINT、MEDIUMINT 和 BIGINT。下面的表显示了需要的每个整数类型的存储和范围。
| 类型 | 大小 |范围（有符号）|范围（无符号）| 用途|
| --- | --- | --- | --- | --- | ---|
|TINYINT|1 Bytes|(-128，127)|(0，255)|小整数值|
|SMALLINT|2 Bytes|(-32 768，32 767)|(0，65 535)|	大整数值|
|MEDIUMINT|3 Bytes|(-8 388 608，8 388 607)|(0，16 777 215)|大整数值|
|INT或INTEGER|4 Bytes|(-2 147 483 648，2 147 483 647)|(0，4 294 967 295)|大整数值|
|BIGINT|8 Bytes|(-3.402 823 466 E+38，-1.175 494 351 E-38)，0，(1.175 494 351 E-38，3.402 823 466 351 E+38)|(0，18 446 744 073 709 551 615)|	极大整数值|
|FLOAT|4 Bytes|(-3.402 823 466 E+38，-1.175 494 351 E-38)|0，(1.175 494 351 E-38，3.402 823 466 E+38)|单精度浮点数值|
|DOUBLE|8 Bytes|(-1.797 693 134 862 315 7 E+308，-2.225 073 858 507 201 4 E-308)，0，(2.225 073 858 507 201 4 E-308，1.797 693 134 862 315 7 E+308)|0，(2.225 073 858 507 201 4 E-308，1.797 693 134 862 315 7 E+308)|双精度浮点数值|
|DECIMAL|对DECIMAL(M,D) ，如果M>D，为M+2否则为D+2|依赖于M和D的值|	依赖于M和D的值|	小数值|

### 3.2 日期和时间类型
表示时间值的日期和时间类型为DATETIME、DATE、TIMESTAMP、TIME和YEAR。
每个时间类型有一个有效值范围和一个"零"值，当指定不合法的MySQL不能表示的值时使用"零"值。
TIMESTAMP类型有专有的自动更新特性，将在后面描述。
| 类型 | 大小(bytes)|范围|格式| 用途|
| --- | --- | --- | --- | --- | ---|
|DATE|3|1000-01-01/9999-12-31|YYYY-MM-DD|日期值|
|TIME|3|'-838:59:59'/'838:59:59'|HH:MM:SS|时间值或持续时间|
|YEAR|1	1901/2155|YYYY|年份值|
|DATETIME|8|1000-01-01|00:00:00/9999-12-31 23:59:59|YYYY-MM-DD|HH:MM:SS|混合日期和时间值|
|TIMESTAMP|4|1970-01-01 00:00:00/2038|结束时间是第 2147483647 秒，北京时间 2038-1-19 11:14:07，格林尼治时间 2038年1月19日 凌晨 03:14:07|YYYYMMDD HHMMSS	混合日期和时间值，时间戳|

### 3.3 字符串类型
字符串类型指CHAR、VARCHAR、BINARY、VARBINARY、BLOB、TEXT、ENUM和SET。该节描述了这些类型如何工作以及如何在查询中使用这些类型。
| 类型 | 大小(bytes)| 用途|
| --- | --- | --- | --- | --- | ---|
|CHAR|0-255|定长字符串|
|VARCHAR|0-65535|变长字符串|
|TINYBLOB|0-255 |不超过 255 个字符的二进制字符串|
|TINYTEXT|	0-255 |	短文本字符串|
|BLOB|	0-65 535 |二进制形式的长文本数据|
|TEXT|	0-65 535| 长文本数据|
|MEDIUMBLOB| 0-16 777 215 |	二进制形式的中等长度文本数据|
|MEDIUMTEXT| 0-16 777 215 |	中等长度文本数据|
|LONGBLOB| 0-4 294 967 295 | 二进制形式的极大文本数据|
|LONGTEXT| 0-4 294 967 295 bytes | 极大文本数据|

注意：char(n) 和 varchar(n) 中括号中 n 代表字符的个数，并不代表字节个数，比如 CHAR(30) 就可以存储 30 个字符。

CHAR 和 VARCHAR 类型类似，但它们保存和检索的方式不同。它们的最大长度和是否尾部空格被保留等方面也不同。在存储或检索过程中不进行大小写转换。

BINARY 和 VARBINARY 类似于 CHAR 和 VARCHAR，不同的是它们包含二进制字符串而不要非二进制字符串。也就是说，它们包含字节字符串而不是字符字符串。这说明它们没有字符集，并且排序和比较基于列值字节的数值值。

BLOB 是一个二进制大对象，可以容纳可变数量的数据。有 4 种 BLOB 类型：TINYBLOB、BLOB、MEDIUMBLOB 和 LONGBLOB。它们区别在于可容纳存储范围不同。

有 4 种 TEXT 类型：TINYTEXT、TEXT、MEDIUMTEXT 和 LONGTEXT。对应的这 4 种 BLOB 类型，可存储的最大长度不同，可根据实际情况选择。

## 4 MySQL管理数据表
### 4.1 创建数据表
创建MySQL数据表需要以下信息：

>- 表名
>- 表字段名
>- 定义每个表字段

#### 语法
以下为创建MySQL数据表的SQL通用语法：
```
CREATE TABLE table_name (column_name column_type);
```
以下例子中我们将在 RUNOOB 数据库中创建数据表runoob_tbl：
```
CREATE TABLE IF NOT EXISTS `runoob_tbl`(
   `runoob_id` INT UNSIGNED AUTO_INCREMENT,
   `runoob_title` VARCHAR(100) NOT NULL,
   `runoob_author` VARCHAR(40) NOT NULL,
   `submission_date` DATE,
   PRIMARY KEY ( `runoob_id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;
```
实例解析：

>- 如果你不想字段为`NULL`可以设置字段的属性为`NOT NULL`， 在操作数据库时如果输入该字段的数据为`NULL`，就会报错。
>- `AUTO_INCREMENT`定义列为自增的属性，一般用于主键，数值会自动加1。
>- `PRIMARY KEY`关键字用于定义列为主键。 您可以使用多列来定义主键，列间以逗号分隔。
>- `ENGINE`设置存储引擎，`CHARSET`设置编码。


#### 通过命令提示符创建表

通过 mysql> 命令窗口可以很简单的创建MySQL数据表。你可以使用 SQL 语句 CREATE TABLE 来创建数据表。
实例:
```
root@host# mysql -u root -p
Enter password:*******
mysql> use RUNOOB;
Database changed
mysql> CREATE TABLE runoob_tbl(
   -> runoob_id INT NOT NULL AUTO_INCREMENT,
   -> runoob_title VARCHAR(100) NOT NULL,
   -> runoob_author VARCHAR(40) NOT NULL,
   -> submission_date DATE,
   -> PRIMARY KEY ( runoob_id )
   -> )ENGINE=InnoDB DEFAULT CHARSET=utf8;
Query OK, 0 rows affected (0.16 sec)
mysql>
```
_注意_：MySQL命令终止符为分号 `;` 。

_注意_： `->`是换行符标识，不要复制。

### 4.2 删除数据表
MySQL中删除数据表是非常容易操作的，但是你在进行删除表操作时要非常小心，因为执行删除命令后所有数据都会消失。
语法
以下为删除MySQL数据表的通用语法：
```
DROP TABLE table_name ;
```

__在命令提示窗口中删除数据表__
在mysql>命令提示窗口中删除数据表SQL语句为`DROP TABLE`：
```
root@host# mysql -u root -p
Enter password:*******
mysql> use RUNOOB;
Database changed
mysql> DROP TABLE runoob_tbl;
Query OK, 0 rows affected (0.8 sec)
mysql>
```
### 4.3 插入数据
MySQL 表中使用`INSERT INTO SQL`语句来插入数据。

你可以通过 mysql> 命令提示窗口中向数据表中插入数据，或者通过PHP脚本来插入数据。

__语法__
以下为向MySQL数据表插入数据通用的`INSERT INTO` SQL语法：
```
INSERT INTO table_name ( field1, field2,...fieldN )
                       VALUES
                       ( value1, value2,...valueN );
```
如果数据是字符型，必须使用单引号或者双引号，如："value"。

__通过命令提示窗口插入数据__
以下我们将使用`SQL INSERT INTO`语句向 MySQL 数据表 `runoob_tbl`插入数据

以下实例中我们将向`runoob_tbl`表插入三条数据:
```
root@host# mysql -u root -p password;
Enter password:*******
mysql> use RUNOOB;
Database changed
mysql> INSERT INTO runoob_tbl 
    -> (runoob_title, runoob_author, submission_date)
    -> VALUES
    -> ("学习 PHP", "菜鸟教程", NOW());
Query OK, 1 rows affected, 1 warnings (0.01 sec)
mysql> INSERT INTO runoob_tbl
    -> (runoob_title, runoob_author, submission_date)
    -> VALUES
    -> ("学习 MySQL", "菜鸟教程", NOW());
Query OK, 1 rows affected, 1 warnings (0.01 sec)
mysql> INSERT INTO runoob_tbl
    -> (runoob_title, runoob_author, submission_date)
    -> VALUES
    -> ("JAVA 教程", "RUNOOB.COM", '2016-05-06');
Query OK, 1 rows affected (0.00 sec)
mysql>
```
在以上实例中，我们并没有提供 runoob_id 的数据，因为该字段我们在创建表的时候已经设置它为 AUTO_INCREMENT(自动增加) 属性。 所以，该字段会自动递增而不需要我们去设置。实例中 NOW() 是一个 MySQL 函数，该函数返回日期和时间。

## 5 查询数据(`SELECT`)

MySQL数据库使用`SQL SELECT`语句来查询数据。

你可以通过`mysql>`命令提示窗口中在数据库中查询数据，或者通过PHP脚本来查询数据。

### 5.1 语法

以下为在MySQL数据库中查询数据通用的`SELECT`语法：
```
SELECT column_name,column_name
FROM table_name
[WHERE Clause]
[LIMIT N][ OFFSET M]
```
>- 查询语句中你可以使用一个或者多个表，表之间使用逗号`(,)`分割，并使用`WHERE`语句来设定查询条件。
>- `SELECT`命令可以读取一条或者多条记录。
>- 你可以使用星号`*`来代替其他字段，`SELECT *`语句会返回表的所有字段数据
>- 你可以使用`WHERE`语句来包含任何条件。
>-你可以使用`LIMIT`属性来设定返回的记录数。
>- 你可以通过`OFFSET`指定`SELECT`语句开始查询的数据偏移量。默认情况下偏移量为`0`。kjj

### 5.2 检索不同的行（`DISTINCT`）
如果需要只返回不同的值，即使每个值只出现一次（去除重复值），我们可以使用 `DISTINCT`关键字
```
SELECT DISTINCT ven_id
FROM products;
```
如果使用`DISTINCT`关键字，它必须直接放在列名的前面

    注意： 不能部分使用`DISTINCT`，它必须置于所有列前面。如果给出`SELECT DISTINCT ven_id, prod_price`，除非指定的两个列都不同，否则所有行都会被检索出来。

### 5.3 限制结果(`LIMIT`)
此语句使用`SELECT`检索单个列。`LIMIT 5`指示返回不多于5行。
```
SELSCT prod_name
FROM products
LIMIT 5;
```
为了得出下一个5行，可指定要检索的开始的行数
```
SELSCT prod_name
FROM products
LIMIT 5，5;
```
`LIMIT 5,5`指示返回从第5行开始的5行。
- 注意：行索引0开始，因此`LIMIT 1,1`将检索出第2行。

### 5.4 使用完全限定的表名
```
SELECT products.prod_name
FROM crashcourse.products;
```
这条语句和不带限定的语句功能完全一样，只是对列名和表名都加了完全限定。

## 6 排序数据
如果不加限定的话，检索出的数据一般将以它在底层表中出现的顺序显示（可以是数据最初添加到表中的顺序）。

### 6.1 按单列排序
为了明确地排序用`SELECT`语句检索出的数据，可使用`ORDER BY`子句，`ORDER BY`子句取一个或多个列的名字，据此对输出排序。
```
SELSCT prod_name
FROM products
ORDER BY prod_name;
```
### 6.2 按多个列排序
为了按多个列排序，只要指定列名，列名之间用逗号分开即可，排序的顺序按照列名的先后顺序进行。下面的例子检索三个列，并按其中两个结果排序——首先按价格，然后再按名称排序：
```
SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY prod_price, prod_name;
```
### 6.3 指定排序方向
升序是默认的排序顺序，为了实现降序排序，必须指定`DESC`关键字。
```
SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY prod_price DESC;
```
- 注意：`DESC`关键字，只应用到直接位于其前面的列名，例如下面的实例只对`prod_price`指定降序，对`prod_name`仍然按升序排序：
```
SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY prod_price DESC, prod_name;
```
与`DESC`相反的关键字是`ASC`，在升序时可以指定它。



## 7 过滤数据
### 7.1 使用`WHERE`子句

我们知道从 MySQL 表中使用`SQL SELECT`语句来读取数据。如需有条件地从表中选取数据，可将`WHERE`子句添加到`SELECT`语句中。


以下是`SQL SELECT`语句使用`WHERE`子句从数据表中读取数据的通用语法：
```
SELECT field1, field2,...fieldN FROM table_name1, table_name2...
[WHERE condition1 [AND [OR]] condition2.....
```
>- 查询语句中你可以使用一个或者多个表，表之间使用逗号, 分割，并使用`WHERE`语句来设定查询条件。
>- 你可以在`WHERE`子句中指定任何条件。
>- 你可以使用`AND`或者`OR`指定一个或多个条件。
>- `WHERE`子句也可以运用于 SQL 的`DELETE`或者`UPDATE`命令。
>- `WHERE`子句类似于程序语言中的`if`条件，根据 MySQL 表中的字段值来读取指定的数据。

- 注意：在同时使用`ORDER BY`和`WHERE`子句是，应该将`ORDER BY`置于`WHERE`之后，否则会产生错误。

### 7.2 `WHERE`子句操作符
|操作符|说明|操作符|说明|
| --- | ---|--- | ---|
| = | 等于 |<=|小于等于|
|<> |不等于|>|大于|
|！= |不等于|>=|大于等于|
|< |小于|BETWEEEN|在范围中|
实例：
```
SELECT peod_name, prod_price
FROM prodcuts
WHERE prod_price < 10;
```

实例：范围检查
```
SELECT peod_name, prod_price
FROM prodcuts
WHERE prod_price BETWEENT 5 AND 10;
```

### 7.3 空值检查
使用`IS NULL`子句检查具有`NULL`值的列
```
SELECT cust_id
FROM customers
WHERE cust_email IS NULL;
```

### 7.4 组合`WHERE`子句

#### `AND`操作符
联结多个条件，指示检索满足搜友给定条件的行。
```
SELECT prod_id, prod_price, prod_name
FROM products
ORDER BY vend_id = 1003 AND prod_price <= 10;
```

#### `OR`操作符
`OR`操作符指示MySQL检索匹配任一条件的行。
```
SELECT prod_id, prod_price
FROM products
ORDER BY vend_id = 1002 OR ven_id = 1002;
```

#### 计算次序（优先级）
`AND`操作符优先级高于`OR`，SQL优先处理`AND`，下面的例子我们的本意是希望等到供应商id为1002 or 1003且 price >= 10的记录：
```
SELECT prod_name, prod_price
FROM products
WHERE vend_id = 1002 OR vend_id = 1003 AND prod_price >= 10;
```
输出结果如下：
```
+----------------+------------+
| prod_name      | prod_price |
+----------------+------------+
| Fuses          |       3.42 |
| Oil can        |       8.99 |
| Detonator      |      13.00 |
| Bird seed      |      10.00 |
| Safe           |      50.00 |
| TNT (5 sticks) |      10.00 |
+----------------+------------+
```
我们可以看到，返回的行中有两行价格小于10，这正是由于`AND`优先级高于`OR`造成的。所以我们需要圆括号明确地分组相应的操作符：
```
SELECT prod_name, prod_price
FROM products
WHERE (vend_id = 1002 OR vend_id = 1003) AND prod_price >= 10;
```
输出结果
```
+----------------+------------+
| prod_name      | prod_price |
+----------------+------------+
| Detonator      |      13.00 |
| Bird seed      |      10.00 |
| Safe           |      50.00 |
| TNT (5 sticks) |      10.00 |
+----------------+------------+
```

####  `IN`操作符
`IN`操作符用来指定条件范围，范围中的每个条件都可以进行匹配。`IN`取合法值的由逗号分隔的清单，全都括在圆括号中。
```
SELECT prod_name, prod_price
FROM products
WHERE vend_id IN (1002, 1003)
ORDER BY prod_name; 
```

#### `NOT`操作符
`WHERE`子句中的`NOT`操作符有且只有一个功能：否定它之后所有的任何条件
```
SELECT prod_name,prod_price 
FROM products 
WHRER vend_id NOT IN (1002,1003) 
ORDER BY prod_name;
```



## 8 用通配符进行过滤
### 8.1 `LIKE`操作符
例子：有时候我们需要获取`runoob_author`字段含有`"COM"`字符的所有记录，这时我们就需要在`WHERE`子句中使用`SQL LIKE`子句。

`SQL LIKE`子句中使用百分号`%`字符来表示任意字符出现任意次数，类似于UNIX或正则表达式中的星号`*`。

#### `%`通配符
如果没有使用百分号`%`, `LIKE`子句与等号`=`的效果是一样的。
以下是`SQL SELECT`语句使用`LIKE`子句从数据表中读取数据的通用语法：
```
SELECT field1, field2,...fieldN 
FROM table_name
WHERE field1 LIKE condition1 [AND [OR]] filed2 = 'somevalue'
```

>- 你可以在`WHERE`子句中指定任何条件。
>- 你可以在`WHERE`子句中使用`LIKE`子句。
>- 你可以使用`LIKE子`句代替等号`=`。
>- `LIKE`通常与`%`一同使用，类似于一个元字符的搜索。
>- 你可以使用`AND`或者`OR`指定一个或多个条件。
>- 你可以在`DELETE`或`UPDATE`命令中使用`WHERE...LIKE`子句来指定条件。

- __注意__: 尾空格可能会干扰通配符。例如，在保存词`anvil`时，如果它后面有一个或多个空格，则子句`LIKE`将不会匹配它们
- __注意__:`%`不能匹配`NULL`。

#### 下划线`_`通配符
下划线只匹配单个字符而不是多个字符
```
SELECT prod_id,prod_name 
FROM products 
WHERE prod_name LIKE '_ ton anvil';
```
输出结果：
```
+---------+-------------+
| prod_id | prod_name   |
+---------+-------------+
| ANV02   | 1 ton anvil |
| ANV03   | 2 ton anvil |
+---------+-------------+
```

#### 8.2 使用正则表达式进行搜索
正则表达式是用来匹配文本的特殊的串（字符集合）。
`.`是正则表达式语言中一个特殊的字符，它表示匹配任意一个字符。
```
SELECT prod_name
FROM products
WHERE prod_name REGEXP '.000'
ORDER BY prod_name;
```
- 注意:
> `LIKE`和正则表达式的区别之一就是——假如被匹配的文本在列值中出现（即列值==被匹配文本），LIKE不会找到它，而`REGEXP`会找到它并返回。
> `BINART`:正则表达式匹配不区分大小写，若要区分大小写，需要指定该关键字，`WHERE prod_name REGEXP BINARY 'JetPack .000'`
> `|`: 为正则表达式的OR操作，如`WHERE prod_name REGEXP '1000|2000'`
> `[]`:匹配括号内的任一字符，如`WHERE prod_name REGEXP '[123] Ton'`。匹配不限于完整的集合，`[1-3]``[a-z]`也是合法的范围.
> 匹配特殊字符：特殊字符使用`\\`转义，如`\\.`

|元字符|说明|定位元字符|说明
| --- |---|---|---|
|`\\f`|换页|`^`|文本的开始|
|`\\n`|换行|`$`|词的开始|
|`\\r`|回车|`[[:<:]]`|词的开始|
|`\\t`|制表|`[[:>:]]`|词的结尾|
|`\\v`|纵向制表|  | |


## 9 创建计算字段
字段基本上与列的意思相同，经常互换使用。

### 9.1 拼接字段
- 拼接： 将值联结到一起构成单个值。

MYSQL使用`Concat()`函数来实现：
```
SELECT CONCAT(vend_name,'(',vend_country,')') 
FROM vendors
ORDER BY vend_name;
```
输出结果：
```
+----------------------------------------+
| CONCAT(vend_name,'(',vend_country,')') |
+----------------------------------------+
| ACME(USA)                              |
| Anvils R Us(USA)                       |
| Furball Inc.(USA)                      |
| Jet Set(England)                       |
| Jouets Et Ours(France)                 |
| LT Supplies(USA)                       |
+----------------------------------------+
```

使用`RTrim(ven_name)删除右侧多余的空格：
```
SELECT CONCAT(RTRIM(vend_name),' (',RTRIM(vend_country),')') 
FROM vendors  
ORDER BY vend_name;
```
- `LTrim()`:去掉串左边的空格

### 9.2 使用别名
```
SELECT CONCAT(vend_name,'(',vend_country,')') AS vend_title
FROM vendors
ORDER BY vend_name;
```
输出结果：
+------------------------+
| vend_title             |
+------------------------+
| ACME(USA)              |
| Anvils R Us(USA)       |
| Furball Inc.(USA)      |
| Jet Set(England)       |
| Jouets Et Ours(France) |
| LT Supplies(USA)       |
+------------------------+

### 9.3 执行算术计算
对检索出的数据进行算术计算
```
SELECT prod_id, quantity, item_price, quantity*item_price AS expanded_price
FROM orderitems
WHERE order_num = 20005;
```
输出结果
+---------+----------+------------+----------------+
| prod_id | quantity | item_price | expanded_price |
+---------+----------+------------+----------------+
| ANV01   |       10 |       5.99 |          59.90 |
| ANV02   |        3 |       9.99 |          29.97 |
| TNT2    |        5 |      10.00 |          50.00 |
| FB      |        1 |      10.00 |          10.00 |
+---------+----------+------------+----------------+


## 10 使用数据处理函数

### 10.1 文本处理函数
常用的文本处理函数：
|函数|说明|函数|说明|
|---|---|---|---|
|`Left()`|返回串左边的字符|`Length()`|返回串的长度|
|`Locate()`|找出串的一个子|`Lower()`|将串转化为小写|
|`LTrim（）`|去掉串左边的空格|`Right()`|返回串右边的字符|
|`RTrim()`|去掉串右边的空格|`Soundex()`|返回串的SOUNDEX值|
|`SubString()`|返回子串的字符|`Upper()`|将串转换为大写|
`Soundex()`是一个将任何文本串转换为描述其语音表示的字母数字模式的算法。下面的例子是匹配发音类似'Y.Lie'的联系名：
```
select cust_name, cust_contact 
from customers
where Soundex(cust_contact) = Soundex('Y. Lie');
```
输出结果：
```
+-------------+--------------+
| cust_name   | cust_contact |
+-------------+--------------+
| Coyote Inc. | Y Lee        |
+-------------+--------------+
```

### 10.2 日期和时间处理函数
下面列出了常用的日期和时间处理函数：
|函数|说明|函数|说明|函数|说明|
|---|---|---|---|---|---|
|`AddDate()`|增加一个日期（天，周） |`AddTime()`|增加一个时间（时，分） |`CurDate()`|返回当前日期|
|`CurTime()`|返回当前时间|`Date()`|返回当前日期时间的日期部分|`DateDiff()`|计算两个日期之差|
|`Date_Add()`|高度灵活的日期运算函数|`Date_Format()`|返回格式化的日期或时间串|
|`Day()`|返回一个日期的天数部分|`DayOfWeek()`|对于一个日期，返回对应的星球几|`Hour()`|返回一个时间的小时部分|
|`Minute()`|返回一个时间的分钟部分|`Mounth()`|返回一个日期的月份部分|
|`Now()`|返回当前的日期和时间|`Second()`|返回一个日期的秒部分|`Time()`|返回一个日期时间的时间部分|
|`Year()`|返回一个日期的年份部分|

- MySQL使用的日期格式有且仅有yyyy-mm-dd
```
select	cust_id, order_num
from orders
where date(order_date) between '2005-09-01' AND '2005-09-30';
```
### 10.3 数值处理函数
|函数|说明|函数|说明|函数|说明|
|---|---|---|---|---|---|
|`Abs()`|返回绝对值|`Cos()`|返回余弦|`Exp()`|返回一个数的指数值|
|`Mod()`|取余|`Pi()`|圆周率|`Rand()`|返回一个随机数|
|`Sin()`|返回一个角度的正弦|`Sqrt()`|返回一个数的平方根|`Tan()`|返回一个角度的正切|



## 11 汇总数据
### 11.1 聚集函数



















### 4.6 `UPDATE`更新
如果我们需要修改或更新 MySQL 中的数据，我们可以使用`SQL UPDATE`命令来操作。

__语法__

以下是`UPDATE`命令修改 MySQL 数据表数据的通用SQL语法：
```
UPDATE table_name SET field1=new-value1, field2=new-value2
[WHERE Clause]
```
>- 你可以同时更新一个或多个字段。
>- 你可以在 WHERE 子句中指定任何条件。
>- 你可以在一个单独表中同时更新数据。

### 4.7 `DELETE`语句
你可以使用 SQL 的`DELETE FROM`命令来删除 MySQL 数据表中的记录。

你可以在`mysql>`命令提示符或 PHP 脚本中执行该命令。
##### 语法
以下是 SQL DELETE 语句从 MySQL 数据表中删除数据的通用语法：
```
DELETE FROM table_name [WHERE Clause]
```
>- 如果没有指定`WHERE`子句，MySQL 表中的所有记录将被删除。
>- 你可以在`WHERE`子句中指定任何条件
>- 您可以在单个表中一次性删除记录。



### 4.9 `UNION`操作符
本教程为大家介绍 MySQL UNION 操作符的语法和实例。

MySQL UNION 操作符用于连接两个以上的`SELECT`语句的结果组合到一个结果集合中。多个`SELECT`语句会删除重复的数据。

##### 语法
MySQL UNION 操作符语法格式：
```
SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions]
UNION [ALL | DISTINCT]
SELECT expression1, expression2, ... expression_n
FROM tables
[WHERE conditions];
```
__参数__
>- `expression1, expression2, ... expression_n:` 要检索的列。
>- `tables:` 要检索的数据表。
>- `WHERE conditions`: 可选， 检索条件。
>- `DISTINCT`: 可选，删除结果集中重复的数据。默认情况下 `UNION` 操作符已经删除了重复数据，所以`DISTINCT`修饰符对结果没啥影响。
>- `ALL`: 可选，返回所有结果集，包含重复数据。