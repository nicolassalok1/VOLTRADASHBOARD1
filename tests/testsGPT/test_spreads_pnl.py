import ast
import json
import math
import tempfile
from pathlib import Path
from unittest import TestCase


def _extract_functions(source_text, func_names):
    tree = ast.parse(source_text)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            segment = ast.get_source_segment(source_text, node)
            if segment:
                segments.append(segment)
    return "\n\n".join(segments)


class OptionsSpreadsPnLTests(TestCase):
    """
    Valide l’ajout et le PnL d’options/spreads via les fonctions de l’app GPT.
    On réutilise add_option_to_dashboard et compute_option_pnl directement depuis streamlit_appGPT.py.
    """

    def setUp(self):
        src = Path("streamlit_appGPT.py").read_text(encoding="utf-8")
        code = "\n".join(
            [
                "import json, time, math",
                "from pathlib import Path",
                _extract_functions(
                    src,
                    {
                        "load_options_book",
                        "save_options_book",
                        "add_option_to_dashboard",
                        "compute_option_payoff",
                        "compute_option_pnl",
                    },
                ),
            ]
        )
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)
        ns = {}
        exec(code, ns)  # nosec: executed from local source

        # Redirige les fichiers JSON vers le répertoire temporaire.
        ns["OPTIONS_BOOK_FILE"] = tmp_path / "options_portfolio.json"
        ns["OPTIONS_BOOK_FILE_LEGACY"] = tmp_path / "options_book.json"
        ns["JSON_DIR"] = tmp_path
        self.api = ns

    def tearDown(self):
        self.tmpdir.cleanup()

    def _add(self, payload):
        return self.api["add_option_to_dashboard"](payload)

    def _book(self):
        return self.api["load_options_book"]()

    def _pnl(self, record, spot, mark=None):
        return self.api["compute_option_pnl"](record, spot_at_event=spot, mark_price=mark)

    def test_vertical_spreads_pnl(self):
        """
        Valide bull call spread et bear put spread :
        - bull call : long K=100 (prime 4), short K=110 (prime 1), S=115 -> payoff 15-5=10, PnL=7
        - bear put : long K=110 (prime 5), short K=100 (prime 2), S=95 -> payoff 15-5=10, PnL=7
        """
        # Bull call spread
        long_call = {
            "id": "bull_long",
            "product_type": "Call spread",
            "option_type": "call",
            "strike": 100,
            "quantity": 1,
            "avg_price": 4.0,
            "side": "long",
        }
        short_call = {
            "id": "bull_short",
            "product_type": "Call spread",
            "option_type": "call",
            "strike": 110,
            "quantity": 1,
            "avg_price": 1.0,
            "side": "short",
        }
        self._add(long_call)
        self._add(short_call)

        # Bear put spread
        long_put = {
            "id": "bear_long",
            "product_type": "Put spread",
            "option_type": "put",
            "strike": 110,
            "quantity": 1,
            "avg_price": 5.0,
            "side": "long",
        }
        short_put = {
            "id": "bear_short",
            "product_type": "Put spread",
            "option_type": "put",
            "strike": 100,
            "quantity": 1,
            "avg_price": 2.0,
            "side": "short",
        }
        self._add(long_put)
        self._add(short_put)

        book = self._book()
        self.assertEqual(len(book), 4)

        # Expiration PnL bull call (S=115)
        bull_long_pnl = self._pnl(book["bull_long"], spot=115.0)
        bull_short_pnl = self._pnl(book["bull_short"], spot=115.0)
        net_bull_pnl = bull_long_pnl["pnl_total"] + bull_short_pnl["pnl_total"]
        self.assertAlmostEqual(net_bull_pnl, 7.0, places=3)

        # Expiration PnL bear put (S=95)
        bear_long_pnl = self._pnl(book["bear_long"], spot=95.0)
        bear_short_pnl = self._pnl(book["bear_short"], spot=95.0)
        net_bear_pnl = bear_long_pnl["pnl_total"] + bear_short_pnl["pnl_total"]
        self.assertAlmostEqual(net_bear_pnl, 7.0, places=3)

    def test_straddle_strangle_close_and_expiry(self):
        """
        Strangle et straddle, avec close (mark_price) et expiration :
        - Strangle long (K_put=95, K_call=105, primes 2 et 3)
        - Straddle long (K=100, primes 4 et 4)
        """
        # Strangle (legs explicites)
        long_put = {
            "id": "strangle_put",
            "product_type": "Strangle",
            "option_type": "put",
            "strike": 95,
            "quantity": 1,
            "avg_price": 2.0,
            "side": "long",
        }
        long_call = {
            "id": "strangle_call",
            "product_type": "Strangle",
            "option_type": "call",
            "strike": 105,
            "quantity": 1,
            "avg_price": 3.0,
            "side": "long",
        }
        self._add(long_put)
        self._add(long_call)

        # Straddle (legs explicites)
        straddle_put = {
            "id": "straddle_put",
            "product_type": "Straddle",
            "option_type": "put",
            "strike": 100,
            "quantity": 1,
            "avg_price": 4.0,
            "side": "long",
        }
        straddle_call = {
            "id": "straddle_call",
            "product_type": "Straddle",
            "option_type": "call",
            "strike": 100,
            "quantity": 1,
            "avg_price": 4.0,
            "side": "long",
        }
        self._add(straddle_put)
        self._add(straddle_call)

        book = self._book()
        # Close mark at S=102 before expiry
        close_put = self._pnl(book["strangle_put"], spot=102.0, mark=0.5)
        close_call = self._pnl(book["strangle_call"], spot=102.0, mark=2.0)
        net_close_strangle = close_put["pnl_total"] + close_call["pnl_total"]
        # Intrinsic at 102: put OTM, call ITM  - premium 5 total => expect negative to small
        self.assertLess(net_close_strangle, 0.0)

        # Expiration at S=90: strangle payoff = put 5, call 0, primes 2+3 => PnL = 5-5=0
        exp_put = self._pnl(book["strangle_put"], spot=90.0)
        exp_call = self._pnl(book["strangle_call"], spot=90.0)
        qty_put = float(book["strangle_put"].get("quantity") or 0.0)
        qty_call = float(book["strangle_call"].get("quantity") or 0.0)

        payoff_put = exp_put["payoff_per_unit"] * qty_put
        payoff_call = exp_call["payoff_per_unit"] * qty_call
        premium_put = float(book["strangle_put"].get("avg_price") or 0.0) * qty_put
        premium_call = float(book["strangle_call"].get("avg_price") or 0.0) * qty_call

        expected_strangle_pnl = (payoff_put - premium_put) + (payoff_call - premium_call)
        net_exp_strangle = exp_put["pnl_total"] + exp_call["pnl_total"]
        self.assertAlmostEqual(net_exp_strangle, expected_strangle_pnl, places=6)

        # Expiration at S=110: straddle payoff call 10, put 0, primes 4+4 => PnL = 10-8=2
        exp_straddle_call = self._pnl(book["straddle_call"], spot=110.0)
        exp_straddle_put = self._pnl(book["straddle_put"], spot=110.0)
        net_exp_straddle = exp_straddle_call["pnl_total"] + exp_straddle_put["pnl_total"]
        qty = float(book["straddle_call"].get("quantity") or 0.0)
        premium_call = float(book["straddle_call"].get("avg_price") or 0.0) * qty
        premium_put = float(book["straddle_put"].get("avg_price") or 0.0) * qty
        payoff_call = exp_straddle_call["payoff_per_unit"] * qty
        payoff_put = exp_straddle_put["payoff_per_unit"] * qty
        expected_straddle_pnl = (payoff_call - premium_call) + (payoff_put - premium_put)
        self.assertAlmostEqual(net_exp_straddle, expected_straddle_pnl, places=6)
