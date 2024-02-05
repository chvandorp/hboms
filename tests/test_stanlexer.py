from hboms import stanlexer

class TestStanLexer:
    def test_find_identifier_strings(self):
        code_str = """
real x = log(3.14);
for ( i in 1:z ) {
    // some comment about x, y and z
    if ( u < 10 ) return some_value_101;
    else catch(22e0);
}
"""
        id_strs = stanlexer.find_identifier_strings(code_str)
        
        expected_id_strs = ["x", "log", "i", "z", "u", "some_value_101", "catch"]
        
        assert id_strs == expected_id_strs
    

    def test_find_used_names(self):
        code_str = """
vector[u] vec1 = [1, 2, 3e1]';
real one = 1.0;
vec_2 ~ normal(3.0, tau^(-one));
/* some comment about vec3 */
// vec_3 ~ laplace(0.0, 1.0);
"""
        names = [
            "vector", "u", "vec", "e", "vec_2", 
            "one", "normal", "vec3", "vec_3"
        ]
        id_strs = stanlexer.find_used_names(code_str, names)
        
        expected_id_strs = ["u", "vec_2", "one", "normal"]
        
        assert id_strs == expected_id_strs