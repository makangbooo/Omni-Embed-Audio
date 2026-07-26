# DPC 远端遗留 `flock` 租约故障说明

请平台管理员协助检查并安全释放以下 DPC 文件上的远端遗留锁租约。请勿删除、
覆盖或修改项目数据。

## 影响

SQuTR DATA-13C 内容校验无法启动。程序使用 non-blocking `flock` 防止两个内容
校验任务同时写入可恢复 manifest/cache；当前旧锁持续返回冲突退出码。

## 资源

- 用户：`jg525`
- 当前实例：`bitahub-a20601801981030400524510`
- 共享挂载：
  `system:/bitahub-member[/b20260317235732695cvaqgr/home/jg525]`
- 文件系统：`dpc`
- 锁文件：
  `/home/jg525/datasets/oea/squtr/manifests/.data13c_content_recovery.lock`

## 已验证事实

- 2026-07-26 20:25:56 +08:00，旧锁 non-blocking `flock` 返回 `73`
  （未取得）。
- 同目录全新 probe 文件的 `flock` 返回 `0`（成功），因此该目录的锁能力正常。
- 当前实例无 `run_data13c_squtr_content_recovery.sh` 或
  `validate_squtr_extracted.py` 进程。
- `lslocks`、`fuser`、`lsof`、`/proc/locks` 与本机开放 FD 均未发现持有者。
- 用户已确认没有其他运行实例挂载 `/home/jg525`。
- 该状态持续数小时，不是短暂竞争。

## 请求

1. 查询该 inode/file lock 对应的 DPC 客户端、租约或已回收实例；
2. 确认没有活跃客户端后，仅释放遗留锁租约；
3. 不删除或修改锁文件、SQuTR 数据、manifest、cache 或日志；
4. 返回处理时间、原持有客户端/实例（若可识别）及释放结果。

项目侧会在平台确认后重新执行 non-blocking `flock` 探针；只有返回 `0` 才会恢复
DATA-13C。
