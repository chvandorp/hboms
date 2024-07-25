import hboms
import pytest
import hboms.stanlang as sl

class TestPluginCode:
    def test_invalid_block_name(self):
        """
        The keys in the plugin code blocks must correspond to valid Stan code block names.
        Otherwise an exception is thrown. Test this.
        """

        plugin_code = {
            "invalid block name" : 'print("sigma = ", sigma)'
        }

        with pytest.raises(Exception):
            hboms.HbomsModel(
                "invalid_plugin_code",
                [hboms.Variable("x")],
                "ddt_x = -a*x;",
                "x_0 = 1.0",
                [hboms.Parameter("a", 0.1, "random"), hboms.Parameter("sigma", 0.1, "fixed")],
                [hboms.Observation("X")],
                [hboms.StanDist("normal", "X", ["x", "sigma"])],
                plugin_code=plugin_code,
                compile_model=False
            )

    def test_code_insertion(self):
        """
        Insert code snippets in all model block and asser that they end up in the right places.
        """

        snippets = [
            "/* this should be at the top of the functions block */",
            "/* this should be at the bottom of the data block */",
            "/* this should be at the bottom of the transformed data block */",
            "/* this should be at the bottom of the parameters block */",
            "/* this should be at the bottom of the transformed parameters block */",
            "/* this should be at the bottom of the model block */",
            "/* this should be at the bottom of the generated quantities block */",
        ]

        plugin_code = dict(zip(sl.model_block_names, snippets))

        model = hboms.HbomsModel(
                "invalid_plugin_code",
                [hboms.Variable("x")],
                "ddt_x = -a*x;",
                "x_0 = 1.0;",
                [hboms.Parameter("a", 0.1, "random"), hboms.Parameter("sigma", 0.1, "fixed")],
                [hboms.Observation("X")],
                [hboms.StanDist("normal", "X", ["x", "sigma"])],
                plugin_code=plugin_code,
                compile_model=False
            )
        
        assert isinstance(model._AST, sl.StanModel)

        stmts = [
            model._AST.functions[1],
            model._AST.data[-1],
            model._AST.transformed_data[-1],
            model._AST.parameters[-1],
            model._AST.transformed_parameters[-1],
            model._AST.model[-1],
            model._AST.generated_quantities[-1],
        ]

        for stmt in stmts:
            assert isinstance(stmt, sl.MixinStmt)

        for snip, stmt in zip(snippets, stmts):
            assert snip == stmt.code



        

