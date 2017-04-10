
using BinDeps
@BinDeps.setup

librsb = library_dependency("librsb")

version = "1.2.0-rc6"
provides(Sources,
    Dict(URI("https://downloads.sourceforge.net/project/librsb/librsb-$version.tar.gz") => librsb))

provides(BuildProcess,
    Dict(Autotools(libtarget="librsb.la",
                   configure_options=["CC=gcc", "FC=gfortran"]) => librsb))

provides(AptGet,
    Dict("librsb-dev" => librsb))

provides(Pacman,
    Dict("librsb" => librsb))


@BinDeps.install Dict(:librsb => :librsb)

