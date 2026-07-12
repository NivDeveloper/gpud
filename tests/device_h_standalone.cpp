// Build-time check: <gpud/Device.h> is the dependency-free public seam
// and must compile standalone under plain C++20 — this TU includes it
// first and includes nothing else.
#include <gpud/Device.h>
