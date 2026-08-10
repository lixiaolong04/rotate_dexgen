"""键盘状态机监听 —— stdin 版(不依赖 pynput / X11)。

为什么不用 pynput: pynput 装的是 X11 全局键盘钩子, 需要能连上 X server(DISPLAY+xhost)。
在 Docker / 换机器 / Wayland / SSH 下钩子挂不上就【静默捕获不到任何键】(q/w/e/t 全失效)。
stdin 版直接从当前终端读键(termios cbreak + select), 不需要 X server, 容器/SSH/Wayland 都能用。
前提: 进程 stdin 是真终端(docker run -it / 直接终端里跑); 若被管道/无 TTY 则打印提示并退出监听。

按键语义(与部署状态机对齐):
    e     -> 抓取(暖启动合拢), 抓稳后保持(freeze), 等你按 Enter
    Enter -> 启动旋转策略
    w     -> 冻结 / 再按继续
    q     -> 沿安全轨迹缓慢张开回家
    t     -> 触觉重标定
    Ctrl-C 仍可急停(cbreak 模式不吞 SIGINT)。
"""
import atexit
import select
import sys
import termios
import threading
import time
import tty

from rl_isaaclab.utils.misc import ThreadSafeValue


class KeyboardListener:
    def __init__(self,
                 deploy_state_flag: ThreadSafeValue,
                 calib_tactile_flag: ThreadSafeValue,
                 hand_ip: str = ""):
        self.deploy_state_flag = deploy_state_flag
        self.calib_tactile_flag = calib_tactile_flag
        self.hand_ip = hand_ip
        self.last_deploy_state_flag = self.deploy_state_flag.get()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._fd = None
        self._old_termios = None

    def _handle_key(self, ch: str):
        '''
        deploy_state_flag:  0=home  1=freeze  2=grasp(reset)
                            3=run policy  4=slow move home
        calib_tactile_flag: 0=silence    1=calibrate tactile
        纯映射(无 I/O), 便于单测。
        '''
        if ch == 'q':
            print('[Keyboard] Slowly moving home.')
            self.deploy_state_flag.set(4)
        elif ch == 'w':
            if self.deploy_state_flag.get() != 1:
                self.last_deploy_state_flag = self.deploy_state_flag.get()
                print('[Keyboard] Freeze actions.')
                self.deploy_state_flag.set(1)
            else:
                # warmup 完成后由 env 把 resume state 设成 3。若尚未抓取，
                # last state 仍为 0，此时保持冻结，避免误按 w 突然张开。
                if self.last_deploy_state_flag == 3:
                    print('[Keyboard] Continue policy.')
                    self.deploy_state_flag.set(3)
                elif self.last_deploy_state_flag == 4:
                    print('[Keyboard] Continue slow home.')
                    self.deploy_state_flag.set(4)
                else:
                    print('[Keyboard] Stay frozen; press e to grasp or Enter to run policy.')
        elif ch == 'e':
            print('[Keyboard] Grasp (warmup). 抓稳后按 Enter 启动策略。')
            # warmup 完成前锁住 Enter；env 在确认轨迹完成后才把 resume
            # state 改为 3，防止抓取过程中误触 Enter 提前运行策略。
            self.last_deploy_state_flag = 0
            self.deploy_state_flag.set(2)
        elif ch in ('\r', '\n'):
            if self.deploy_state_flag.get() == 1 and self.last_deploy_state_flag == 3:
                print('[Keyboard] Start policy.')
                self.deploy_state_flag.set(3)
            elif self.deploy_state_flag.get() != 3:
                print('[Keyboard] Enter ignored: finish warmup with e first.')
        elif ch == 't':
            self.calib_tactile_flag.set(1)
            print("[Keyboard] Tactile calibration.")
        else:
            pass

    def _restore(self):
        if self._fd is not None and self._old_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def _run(self):
        try:
            self._fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self._fd)
        except Exception as e:  # 无 TTY(管道/后台/无终端) -> 无法读键
            print(f"[Keyboard] stdin 不是终端, 键盘监听不可用 ({e!r})。"
                  " 请在 `docker run -it ...` 的交互终端里跑, 或用 --no_keyboard 自动跑。")
            return
        atexit.register(self._restore)   # 兜底: 进程异常退出也恢复终端
        try:
            tty.setcbreak(self._fd)       # cbreak: 单键即时读, 无需回车; 保留 Ctrl-C 急停
            while not self.stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    self._handle_key(ch)
        finally:
            self._restore()
        print("[Keyboard] Keyboard listener stopped.")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self._restore()
