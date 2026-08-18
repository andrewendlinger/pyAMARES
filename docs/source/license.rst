License
=======

License for the pyAMARES project:

.. literalinclude:: ../../LICENSE.txt
   :language: text
   :caption: BSD 3-Clause License

Third-Party Licenses
--------------------

This pyAMARES project uses third-party libraries. Below is information about their licenses.

hlsvdpro
^^^^^^^^
The library `hlsvdpro` is used under its BSD-3-Clause License. For detailed license information, please visit the `hlsvdpro PyPI page <https://pypi.org/project/hlsvdpro/>`_.

hlsvdpropy (vendored)
^^^^^^^^^^^^^^^^^^^^^

``pyAMARES/libs/hlsvd.py`` is a vendored copy of ``hlsvdpropy/hlsvd.py`` from
`hlsvdpropy <https://github.com/bsoher/hlsvdpropy>`_ version 2.0.2, Copyright (c) 2020 Brian J Soher, used under its BSD 3-Clause License.
It is the fallback HSVD backend, used whenever the compiled ``hlsvdpro`` package is unavailable — always under NumPy 2.x, and on any platform for which ``hlsvdpro`` ships no wheel.
The full license text, and the list of local modifications, are reproduced in the header of that file.

Additional Note on MPFIR Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MPFIR function within pyAMARES is inspired by MPFIR function in the Matlab software `SPID <https://homes.esat.kuleuven.be/~sistawww/biomed/etumour/SPID/ManualSPID.pdf>`_, which at the time of this implementation had no clear licensing information available. 
It is important to note that ``pyAMARES.libs.MPFIR`` is an independent implementation developed in Python and does not contain any original SPID code.

This function is included in pyAMARES under the same BSD 3-Clause License, and no claim is made on the original SPID software or its intellectual property. 
Users are advised to ensure their use of the MPFIR function complies with legal and regulatory requirements.

Disclaimer
^^^^^^^^^^

pyAMARES and its MPFIR function are not endorsed by or affiliated with SPID or its creators.


