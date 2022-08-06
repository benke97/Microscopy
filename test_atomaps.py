%matplotlib notebook
import hyperspy.api as hs
import atomap.api as am
s = am.dummy_data.get_fantasite()
s.plot()