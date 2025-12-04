from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
import sys
import math
import re

# Functions recognized by parser
FUNCTIONS = {"sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "ln", "log"}

# -----------------------------
# AUTO-CLOSE BRACKETS & FUNCTIONS
# -----------------------------
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
        expr += closing_for[stack.pop()]
    return expr

# -----------------------------
# TOKENIZER & PARSER (Shunting-Yard)
# -----------------------------
token_re = re.compile(
    r"\s*(?:(\d+(?:\.\d*)?|\.\d+)|([A-Za-z_]\w*)|(\*\*|\^|[+\-*/()%(),]))"
)

def tokenize(expr: str):
    pos = 0
    length = len(expr)
    while pos < length:
        m = token_re.match(expr, pos)
        if not m:
            raise ValueError(f"Invalid token at: {expr[pos:]}")
        num, ident, op = m.groups()
        pos = m.end()
        if num:
            yield ("num", float(num))
        elif ident:
            yield ("id", ident)
        else:
            if op == ",":
                yield ("comma", ",")
            else:
                yield ("op", op)
    yield ("end", None)

OP_INFO = {
    "+": (2, "L"),
    "-": (2, "L"),
    "*": (3, "L"),
    "/": (3, "L"),
    "^": (4, "R"),
    "**": (4, "R"),
    "%": (5, "P"),
}

def shunting_yard(tokens):
    output = []
    stack = []
    prev_token_type = None

    for ttype, val in tokens:
        if ttype == "num":
            output.append(("num", val))
            prev_token_type = "num"
        elif ttype == "id":
            if val in FUNCTIONS:
                stack.append(("func", val))
            else:
                output.append(("id", val))
            prev_token_type = "id"
        elif ttype == "op":
            op = val
            if op == "(":
                stack.append(("op", "("))
            elif op == ")":
                while stack and stack[-1][1] != "(":
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Mismatched parentheses")
                stack.pop()
                if stack and stack[-1][0] == "func":
                    output.append(stack.pop())
            else:
                info = OP_INFO.get(op)
                if info is None:
                    raise ValueError(f"Unknown operator: {op}")
                prec, assoc = info
                if assoc == "P":
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
            while stack and stack[-1][1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("Misplaced comma")
            prev_token_type = "comma"
        elif ttype == "end":
            break
    while stack:
        top = stack.pop()
        if top[1] in ("(", ")"):
            raise ValueError("Mismatched parentheses")
        output.append(top)
    return output

# -----------------------------
# RPN EVALUATOR
# -----------------------------
def make_function_map(deg_mode: bool):
    def wrap_forward(fn):
        return lambda x: fn(x if not deg_mode else (x * math.pi / 180.0))
    def wrap_inverse(fn):
        return lambda x: (fn(x) * 180.0 / math.pi) if deg_mode else fn(x)
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
            elif val == "%":
                a = stack.pop(); stack.append(a / 100.0)
        elif ttype == "func":
            a = stack.pop()
            stack.append(fmap[val](a))
    if len(stack) != 1:
        raise ValueError("Malformed expression")
    return stack[0]

def build_and_eval(internal_expr: str, deg_mode: bool):
    tokens = list(tokenize(internal_expr))
    rpn = shunting_yard(tokens)
    return eval_rpn(rpn, deg_mode=deg_mode)

# -----------------------------
# Calculator App
# -----------------------------
class Calculator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.display_current = ""
        self.internal_current = ""
        self.full_internal = ""
        self.full_display = ""
        self.expectingDeg = False

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

        # Functions
        for fname, btn_name in [("sin","btnSin"),("cos","btnCos"),("tan","btnTan"),
                                ("asin","btnAsin"),("acos","btnAcos"),("atan","btnAtan"),
                                ("log","btnLog"),("ln","btnLn")]:
            btn = self.ui.findChild(QPushButton, btn_name)
            if btn:
                btn.clicked.connect(lambda _, f=fname: self.handle_func(f))

        # Actions
        self.ui.btnDegRad.clicked.connect(self.changeAngleUnit)
        self.ui.btnEquals.clicked.connect(self.calculate)
        self.ui.btnClear.clicked.connect(self.clear_expression)
        self.ui.btnBackspace.clicked.connect(self.backspace)

        self.updateAngleButton()
        self.ui.display.setText("0")

    # -----------------------------
    # Input handlers
    # -----------------------------
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
        self.full_internal += op
        disp_op = op.replace("*","\u00d7").replace("/","\u00f7")
        self.full_display += disp_op
        self.display_current = ""
        self.internal_current = ""
        self.ui.display.setText(self.full_display)

    def handle_special(self, ch: str):
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
        self.display_current += f"{fname}("
        self.internal_current += f"{fname}("
        self.ui.display.setText(self.display_current)

    def clear_expression(self):
        self.display_current = ""
        self.internal_current = ""
        self.full_internal = ""
        self.full_display = ""
        self.ui.display.setText("0")
        self.ui.history.setText("")

    def backspace(self):
        if self.display_current:
            self.display_current = self.display_current[:-1]
            temp = self.display_current
            temp = temp.replace("\u03C0","pi").replace("\u221A(","sqrt(")
            self.internal_current = temp
            self.ui.display.setText(self.display_current if self.display_current else "0")
        elif self.full_display:
            self.full_display = self.full_display[:-1]
            temp = self.full_display
            temp = temp.replace("\u03C0","pi").replace("\u221A(","sqrt(").replace("\u00d7","*").replace("\u00f7","/")
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
            factor = 10**14
            rounded = math.ceil(result*factor)/factor
            self.ui.display.setText(str(rounded))
            self.ui.history.setText(str(result))
            self.display_current = str(rounded)
            self.internal_current = str(rounded)
            self.full_internal = ""
            self.full_display = ""
        except Exception as e:
            print("Evaluation error:", e)
            self.ui.display.setText("Error")
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
