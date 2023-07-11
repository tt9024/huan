rm -f td_parser_module_np.cpython*-linux-gnu.so
rm -fR build
python3 setup.py build_ext --inplace

