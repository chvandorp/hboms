from hboms import utilities as util


class TestUtilities:
    def test_flatten(self):
        xss = [[1,2,3], [4,5], [6], []]
        
        assert util.flatten(xss) == [1,2,3,4,5,6]

    def test_unique(self):
        xs = [1,2,2,3,1,5]
        
        assert util.unique(xs) == [1,2,3,5]
        
        xs = [1,5,4,3,4,2,1]
        
        assert util.unique(xs) == [1,2,3,4,5]
        
    def test_find_used_names(self):
        code_str = "a * b.0 + func(2e1 + my_var)"
        
        names = util.find_used_names(code_str, ["a", "b", "e", "my_var"])
        
        expected_names = ["a", "b", "my_var"]
        
        assert expected_names == names
        
        code_str = "if ( x < 10 ) { return (1.0 + 2.0i) / 1.1e-3;"
        
        names = util.find_used_names(code_str, ["x", "i", "e", "urn"])
        
        expected_names = ["x"]
        
        assert expected_names == names