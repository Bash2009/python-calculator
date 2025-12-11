from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
import sys
import math
import re


# Function set for parser/evaluator
FUNCTIONS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "ln", "log",
    "max", "min", "root", "mod", "fact"
}


# AUTO-CLOSE BRACKETS & FUNCTIONS
def auto_close(expr: str) -> str:
    stack = []
    closing_for = {'(': ')', '[': ']', '{': '}'}

    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isalpha():
            match = re.match(r"[a-zA-Z_]\w*", expr[i:])
            if match:
                word = match.group(0)
                end = i + len(word)
                if word in FUNCTIONS and end < len(expr) and expr[end] == "(":
                    stack.append("(")
                    i = end + 1
                    continue

        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if stack:
                stack.pop()
        i += 1

    while stack:
        opening = stack.pop()
        expr += closing_for[opening]
    return expr


# TOKENIZER & PARSER (Shunting-Yard) with unary, postfix, and multi-arg support
token_re = re.compile(
    r"\s*(?:(\d+(?:\.\d*)?|\.\d+)|([A-Za-z_]\w*)|(\*\*|\^|[+\-*/()%(),!]))"
)


def tokenize(expr: str):
    """Tokenize while distinguishing unary minus (u-) from binary minus."""
    pos = 0
    length = len(expr)
    last_type = None  # None, 'num', 'id', 'op', 'comma', '('
    while pos < length:
        m = token_re.match(expr, pos)
        if not m:
            raise ValueError(f"Invalid token at: {expr[pos:]}")
        num, ident, op = m.groups()
        pos = m.end()
        if num:
            yield ("num", float(num))
            last_type = "num"
        elif ident:
            yield ("id", ident)
            last_type = "id"
        else:
            # distinguish unary minus
            if op == "-":
                if last_type in (None, "op", "comma", "("):
                    yield ("op", "u-")
                else:
                    yield ("op", "-")
                last_type = "op"
            elif op == "(":
                yield ("op", "(")
                last_type = "("
            elif op == ")":
                yield ("op", ")")
                last_type = "op"
            elif op == ",":
                yield ("comma", ",")
                last_type = "comma"
            else:
                yield ("op", op)
                last_type = "op"
    yield ("end", None)


OP_INFO = {
    "+": (2, "L"),
    "-": (2, "L"),
    "*": (3, "L"),
    "/": (3, "L"),
    "^": (4, "R"),
    "**": (4, "R"),
    "u-": (5, "R"),   # unary minus
    "!": (6, "P"),    # factorial (postfix)
    "%": (6, "P"),    # percent postfix (existing behavior)
    # 'P' means postfix/immediate-handling (we treat by immediately emitting)
}


def shunting_yard(tokens):
    """
    Produces RPN. Handles:
      - functions with multiple arguments (tracks arg counts)
      - unary minus (u-)
      - postfix operators like ! and %
      - '|' as absolute value delimiter producing an 'abs' function around the content
    RPN uses:
      - ("num", value)
      - ("id", name)
      - ("op", symbol)
      - ("func", (name, argc))  # argc is number of args for multi-arg functions
    """
    output = []
    stack = []
    func_arg_stack = []  # counts commas/arguments for each function encountered
    prev_token_type = None

    for ttype, val in tokens:
        if ttype == "num":
            output.append(("num", val))
            prev_token_type = "num"
        elif ttype == "id":
            # identifier: variable or function
            if val in FUNCTIONS:
                # push function marker and prepare to count its args when '(' arrives
                stack.append(("func", val))
                func_arg_stack.append(0)
            else:
                output.append(("id", val))
            prev_token_type = "id"
        elif ttype == "op":
            op = val

            if op == "(":
                stack.append(("op", "("))
                prev_token_type = "("
            elif op == ")":
                while stack and stack[-1][1] != "(":
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Mismatched parentheses")
                stack.pop()  # pop "("
                # If there is a function on top of stack, pop it and emit with argcount
                if stack and stack[-1][0] == "func":
                    func_name = stack.pop()[1]
                    # determine argument count: if func_arg_stack exists, use it; else assume 1
                    if func_arg_stack:
                        argc = func_arg_stack.pop() + 1
                    else:
                        argc = 1
                    output.append(("func", (func_name, argc)))
            else:
                # operators and postfix tokens
                info = OP_INFO.get(op)
                if info is None:
                    # unknown operator - treat as binary but error
                    raise ValueError(f"Unknown operator: {op}")
                prec, assoc = info
                if assoc == "P":
                    # immediate postfix operators are emitted directly as op tokens
                    output.append(("op", op))
                else:
                    while stack and stack[-1][0] == "op":
                        top = stack[-1][1]
                        if top == "(":
                            break
                        top_info = OP_INFO.get(top)
                        if not top_info:
                            break
                        top_prec, top_assoc = top_info
                        if (top_prec > prec) or (top_prec == prec and assoc == "L"):
                            output.append(stack.pop())
                        else:
                            break
                    stack.append(("op", op))
                prev_token_type = "op"
        elif ttype == "comma":
            # comma separates function arguments; pop until '('
            while stack and stack[-1][1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("Misplaced comma or mismatched parentheses")
            # increment argument count for the current function
            if not func_arg_stack:
                raise ValueError("Comma outside of function")
            func_arg_stack[-1] += 1
            prev_token_type = "comma"
        elif ttype == "end":
            break

    # empty remaining stack
    while stack:
        top = stack.pop()
        if top[1] in ("(", ")",
                      "|"):  # '|' should never remain if matched; parentheses mismatch otherwise
            raise ValueError("Mismatched parentheses or '|'")
        output.append(top)
    return output


# RPN EVALUATOR with multi-arg functions, factorial, unary minus, mod, root
def make_function_map(deg_mode: bool):
    def wrap_forward(fn):
        return lambda x: fn(x if not deg_mode else (x * math.pi / 180.0))

    def wrap_inverse(fn):
        return lambda x: (fn(x) * 180.0 / math.pi) if deg_mode else fn(x)

    def nth_root(n, x):
        # root(n, x) -> x ** (1/n), guard negative roots appropriately
        if n == 0:
            raise ValueError("root: zero-degree")
        return x ** (1.0 / n)

    fmap = {
        "sin": wrap_forward(math.sin),
        "cos": wrap_forward(math.cos),
        "tan": wrap_forward(math.tan),
        "asin": wrap_inverse(math.asin),
        "acos": wrap_inverse(math.acos),
        "atan": wrap_inverse(math.atan),
        "sqrt": lambda x: math.sqrt(x),
        "ln": lambda x: math.log(x),
        "log": lambda x: math.log10(x),
        "neg": lambda x: -x,
        "max": lambda *args: max(*args),
        "min": lambda *args: min(*args),
        "root": lambda n, x: nth_root(n, x),
        "mod": lambda a, b: a % b,
        "fact": lambda x: math.factorial(int(x)) if float(x).is_integer() and x >= 0 else math.gamma(x + 1),
    }
    return fmap


def eval_rpn(rpn, deg_mode=False):
    stack = []
    fmap = make_function_map(deg_mode)
    for token in rpn:
        ttype, val = token
        if ttype == "num":
            stack.append(val)
        elif ttype == "id":
            if val == "pi":
                stack.append(math.pi)
            elif val == "e":
                stack.append(math.e)
            else:
                raise ValueError(f"Unknown identifier {val}")
        elif ttype == "op":
            if val == "+":
                b = stack.pop(); a = stack.pop(); stack.append(a + b)
            elif val == "-":
                b = stack.pop(); a = stack.pop(); stack.append(a - b)
            elif val == "*":
                b = stack.pop(); a = stack.pop(); stack.append(a * b)
            elif val == "/":
                b = stack.pop(); a = stack.pop(); stack.append(a / b)
            elif val in ("^", "**"):
                b = stack.pop(); a = stack.pop(); stack.append(a ** b)
            elif val == "u-":
                a = stack.pop(); stack.append(-a)
            elif val == "%":
                a = stack.pop(); stack.append(a / 100.0)
            elif val == "!":
                a = stack.pop()
                # factorial: for integers use math.factorial, otherwise use gamma
                if float(a).is_integer() and a >= 0:
                    stack.append(float(math.factorial(int(a))))
                else:
                    stack.append(math.gamma(a + 1))
            else:
                raise ValueError(f"Unhandled operator in evaluator: {val}")
        elif ttype == "func":
            # val can be ("name", argc) if multi-arg, or just name in some cases
            if isinstance(val, tuple):
                fname, argc = val
                if argc < 0:
                    raise ValueError("Function argument count invalid")
                if len(stack) < argc:
                    raise ValueError(f"Not enough arguments for function {fname}")
                # pop args in reverse and call
                args = [stack.pop() for _ in range(argc)][::-1]
                func = fmap.get(fname)
                if func is None:
                    raise ValueError(f"Unknown function {fname}")
                # support variable arity
                result = func(*args)
                stack.append(result)
            else:
                # single-arg function represented directly (backwards compatibility)
                fname = val
                a = stack.pop()
                func = fmap.get(fname)
                if func is None:
                    raise ValueError(f"Unknown function {fname}")
                stack.append(func(a))
    if len(stack) != 1:
        raise ValueError("Malformed expression after evaluation")
    return stack[0]


def build_and_eval(internal_expr: str, deg_mode: bool):
    tokens = list(tokenize(internal_expr))
    rpn = shunting_yard(tokens)
    return eval_rpn(rpn, deg_mode=deg_mode)


# Calculator App Class
class Calculator(QMainWindow):
    def __init__(self):
        super().__init__()

        # State variables
        self.display_current = ""
        self.internal_current = ""
        self.full_internal = ""
        self.full_display = ""
        self.expectingDeg = False
        self.prev_answer = ""

        loader = QUiLoader()
        file = QFile("sci-calculator.ui")
        if not file.open(QFile.ReadOnly):
            print("Cannot open sci-calculator.ui")
            sys.exit(-1)
        loaded_ui = loader.load(file, self)
        file.close()
        self.ui = loaded_ui
        self.setCentralWidget(loaded_ui)
        self.setWindowTitle("Scifi Calculator")

        # Numbers
        for i in range(10):
            btn = self.ui.findChild(QPushButton, f"btn{i}")
            if btn:
                btn.clicked.connect(lambda _, d=str(i): self.handle_digit(d))

        # Operators
        self.ui.btnAdd.clicked.connect(lambda _: self.handle_operator("+"))
        self.ui.btnSub.clicked.connect(lambda _: self.handle_operator("-"))
        self.ui.btnMul.clicked.connect(lambda _: self.handle_operator("*"))
        self.ui.btnDiv.clicked.connect(lambda _: self.handle_operator("/"))

        # Special
        self.ui.btnLeftParen.clicked.connect(lambda _: self.handle_special("("))
        self.ui.btnRightParen.clicked.connect(lambda _: self.handle_special(")"))
        self.ui.btnDot.clicked.connect(lambda _: self.handle_dot())
        self.ui.btnPercent.clicked.connect(lambda _: self.handle_percent())
        self.ui.btnPi.clicked.connect(lambda _: self.handle_pi())
        self.ui.btnExp.clicked.connect(lambda _: self.handle_exp())
        self.ui.btnPow.clicked.connect(lambda _: self.handle_operator("^"))
        self.ui.btnInv.clicked.connect(lambda _: self.handle_operator("^") or self.handle_digit("-1"))
        self.ui.btnSqrt.clicked.connect(lambda _: self.handle_sqrt())
        self.ui.btnFact.clicked.connect(lambda _: self.handle_operator("!"))
        self.ui.btnAlog.clicked.connect(lambda _: self.handle_digit("10^"))

        function_binds = [("sin", "btnSin"), ("cos", "btnCos"), ("tan", "btnTan"),
                                ("asin", "btnAsin"), ("acos", "btnAcos"), ("atan", "btnAtan"),
                                ("log", "btnLog"), ("ln", "btnLn"),
                                ("max", "btnMax"), ("min", "btnMin"), ("root", "btnRoot"),
                                ("fact", "btnFact"), ("mod", "btnMod")]

        # Functions
        for fname, btn_name in function_binds:
            btn = self.ui.findChild(QPushButton, btn_name)
            if btn:
                btn.clicked.connect(lambda _, f=fname: self.handle_func(f))

        # Actions
        self.ui.btnDegRad.clicked.connect(self.changeAngleUnit)
        self.ui.btnEquals.clicked.connect(self.calculate)
        self.ui.btnClear.clicked.connect(self.clear_expression)
        self.ui.btnBackspace.clicked.connect(self.backspace)
        self.ui.btnAns.clicked.connect(lambda _: self.insert_ans())

        self.updateAngleButton()
        self.ui.display.setText("0")


    # Input handlers
    def handle_digit(self, d: str):
        self.display_current += d
        self.internal_current += d
        self.ui.display.setText(self.display_current)

    def handle_dot(self):
        m = re.search(r"(\d*\.\d*|\d+)$", self.internal_current)
        if m and "." in m.group(0):
            return
        if not self.internal_current or not re.search(r"\d$", self.internal_current):
            self.display_current += "0."
            self.internal_current += "0."
        else:
            self.display_current += "."
            self.internal_current += "."
        self.ui.display.setText(self.display_current)

    def handle_operator(self, op: str):
        if self.internal_current:
            self.full_internal += self.internal_current
            self.full_display += self.display_current
        # For postfix operators like '!' we must attach directly
        if op in ("!", "%"):
            # attach to the most recent number/expression
            if self.internal_current:
                self.internal_current += op
                self.display_current += op
            elif self.full_internal:
                self.full_internal += op
                self.full_display += op
        else:
            self.full_internal += op
            disp_op = op.replace("*", "\u00d7").replace("/", "\u00f7").replace("^", "^")
            self.full_display += disp_op
            self.display_current = ""
            self.internal_current = ""
        self.ui.display.setText(self.display_current if self.display_current else self.full_display)

    def handle_special(self, ch: str):
        # handles parentheses
        self.display_current += ch
        self.internal_current += ch
        self.ui.display.setText(self.display_current)

    def handle_percent(self):
        if self.internal_current:
            self.internal_current += "%"
            self.display_current += "%"
        elif self.full_internal:
            self.full_internal += "%"
            self.full_display += "%"
        self.ui.display.setText(self.display_current if self.internal_current else self.full_display)

    def handle_pi(self):
        self.display_current += "\u03C0"
        self.internal_current += "pi"
        self.ui.display.setText(self.display_current)

    def handle_exp(self):
        self.display_current += "e^("
        self.internal_current += "e**("
        self.ui.display.setText(self.display_current)

    def handle_sqrt(self):
        self.display_current += "\u221A("
        self.internal_current += "sqrt("
        self.ui.display.setText(self.display_current)

    def handle_func(self, fname: str):
        # For functions that take multiple args, insert name and open parenthesis
        self.display_current += f"{fname}("
        self.internal_current += f"{fname}("
        self.ui.display.setText(self.display_current)

    def insert_ans(self):
        # Insert the last evaluated answer from history into the current input
        hist_text = self.prev_answer
        if hist_text:
            # try parse float from history, otherwise insert as-is
            try:
                _ = float(hist_text)
                self.display_current += hist_text
                self.internal_current += hist_text
                self.ui.display.setText(self.display_current)
            except Exception:
                # if history contains a more complex thing, ignore
                pass

    def clear_expression(self):
        self.display_current = ""
        self.internal_current = ""
        self.full_internal = ""
        self.full_display = ""
        self.ui.display.setText("0")

    def backspace(self):
        if self.display_current:
            self.display_current = self.display_current[:-1]
            temp = self.display_current
            temp = temp.replace("\u03C0", "pi").replace("\u221A(", "sqrt(")
            self.internal_current = temp
            self.ui.display.setText(self.display_current if self.display_current else "0")
        elif self.full_display:
            self.full_display = self.full_display[:-1]
            temp = self.full_display
            temp = temp.replace("\u03C0", "pi").replace("\u221A(", "sqrt(").replace("\u00d7", "*").replace("\u00f7", "/")
            self.full_internal = temp
            self.ui.display.setText(self.full_display if self.full_display else "0")
        else:
            self.ui.display.setText("0")

    def changeAngleUnit(self):
        self.expectingDeg = not self.expectingDeg
        self.updateAngleButton()

    def updateAngleButton(self):
        self.ui.btnDegRad.setText("DEG" if self.expectingDeg else "RAD")

    def calculate(self):
        if not self.internal_current and not self.full_internal:
            return
        expr_internal = self.full_internal + self.internal_current
        expr_internal = auto_close(expr_internal)
        try:
            result = build_and_eval(expr_internal, deg_mode=self.expectingDeg)
            factor = 10 ** 14
            rounded = math.ceil(result * factor) / factor
            self.ui.display.setText(str(rounded))
            if hasattr(self.ui, "history"):
                self.ui.history.setText(str(result))
            self.display_current = str(rounded)
            self.internal_current = str(rounded)
            self.prev_answer = str(rounded)
            self.full_internal = ""
            self.full_display = ""
        except Exception as e:
            print("Evaluation error:", e)
            self.ui.display.setText("Error")
            if hasattr(self.ui, "history"):
                self.ui.history.setText("")
            self.display_current = ""
            self.internal_current = ""
            self.full_internal = ""
            self.full_display = ""


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Calculator()
    window.show()
    sys.exit(app.exec())
