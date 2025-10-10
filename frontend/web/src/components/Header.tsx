import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { Menu, X, ChevronDown } from "lucide-react";

export default function Header() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const closeTimer = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navItems = [
    { label: "Home", href: "/", active: true },
    {
      label: "AI Features",
      dropdown: [
        { label: "Crop Monitoring", href: "/crop-monitoring" },
        { label: "Planting Recommendations", href: "/planting-recommendations" },
        { label: "Harvest Readiness", href: "/harvest-readiness" },
      ],
    },
    { label: "About Us", href: "/about" },
  ];

  const handleMouseEnter = (label: string) => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    setOpenDropdown(label);
  };

  const handleMouseLeave = () => {
    closeTimer.current = setTimeout(() => setOpenDropdown(null), 200);
  };

  const toggleDropdown = (label: string) => {
    setOpenDropdown(openDropdown === label ? null : label);
  };

  return (
    <header
      className={`fixed top-6 left-1/2 -translate-x-1/2 z-50 w-[92%] max-w-6xl transition-all duration-500 border rounded-full
      ${
        scrolled
          ? "bg-white/90 border-gray-200 text-gray-800 shadow-lg backdrop-blur-xl"
          : "bg-white/10 border-white/20 text-white backdrop-blur-md"
      }`}
    >
      <div className="flex items-center justify-between px-6 py-3">
        {/* Logo */}
        <Link to="/" className="font-extrabold text-xl tracking-tight">
          AgriFinSight
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden md:flex gap-6 text-sm font-medium relative">
          {navItems.map((item) => (
            <div
              key={item.label}
              className="relative group"
              onMouseEnter={() => item.dropdown && handleMouseEnter(item.label)}
              onMouseLeave={() => item.dropdown && handleMouseLeave()}
            >
              {item.dropdown ? (
                <button
                  className="flex items-center gap-1 transition hover:opacity-80"
                  onClick={() => toggleDropdown(item.label)}
                >
                  {item.label}
                  <ChevronDown
                    size={14}
                    className={`transition-transform duration-300 ${
                      openDropdown === item.label ? "rotate-180" : ""
                    }`}
                  />
                </button>
              ) : (
                <Link
                  to={item.href}
                  className={`transition hover:opacity-80 ${
                    item.active ? "font-semibold" : ""
                  }`}
                >
                  {item.label}
                </Link>
              )}

              {/* Dropdown */}
              {item.dropdown && openDropdown === item.label && (
                <div
                  onMouseEnter={() => handleMouseEnter(item.label)}
                  onMouseLeave={handleMouseLeave}
                  className={`absolute left-0 mt-3 rounded-xl border shadow-lg py-2 min-w-[200px] 
                    ${
                      scrolled
                        ? "bg-white text-gray-700 border-gray-200"
                        : "bg-white/95 text-gray-800 border-white/20"
                    }`}
                >
                  {item.dropdown.map((sub) => (
                    <Link
                      key={sub.label}
                      to={sub.href}
                      className="block px-4 py-2 hover:bg-gray-100 transition rounded-md"
                    >
                      {sub.label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          ))}
        </nav>

        {/* Desktop Buttons */}
        <div className="hidden md:flex gap-3">
          <button
            className={`px-4 py-2 rounded-full border text-sm font-medium transition
            ${
              scrolled
                ? "border-gray-300 hover:bg-gray-100 text-gray-800"
                : "border-white/30 hover:bg-white/10 text-white"
            }`}
          >
            Contact us
          </button>
          <button
            className={`px-4 py-2 rounded-full text-sm font-semibold transition
            ${
              scrolled
                ? "bg-blue-600 text-white hover:bg-blue-700"
                : "bg-white text-blue-700 hover:bg-blue-100"
            }`}
          >
            Sign In
          </button>
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="md:hidden flex items-center justify-center p-2 rounded-lg focus:outline-none"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile Drawer */}
      <div
        className={`md:hidden transition-all duration-300 overflow-hidden ${
          menuOpen ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"
        }`}
      >
        <nav
          className={`flex flex-col px-6 py-4 space-y-3 border-t ${
            scrolled ? "bg-white text-gray-800" : "bg-white/95 text-gray-800"
          }`}
        >
          {navItems.map((item) => (
            <div key={item.label}>
              <button
                onClick={() =>
                  item.dropdown ? toggleDropdown(item.label) : setMenuOpen(false)
                }
                className="flex items-center justify-between w-full text-left text-sm font-medium"
              >
                {item.label}
                {item.dropdown && (
                  <ChevronDown
                    size={16}
                    className={`transition-transform duration-300 ${
                      openDropdown === item.label ? "rotate-180" : ""
                    }`}
                  />
                )}
              </button>

              {item.dropdown && openDropdown === item.label && (
                <div className="mt-2 ml-3 flex flex-col space-y-2">
                  {item.dropdown.map((sub) => (
                    <Link
                      key={sub.label}
                      to={sub.href}
                      className="text-sm text-gray-600 hover:text-gray-900 transition"
                      onClick={() => setMenuOpen(false)}
                    >
                      {sub.label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Mobile Buttons */}
          <div className="border-t pt-4 mt-4 flex flex-col gap-3">
            <button className="w-full border rounded-full py-2 text-sm hover:bg-gray-100">
              Contact us
            </button>
            <button className="w-full bg-blue-600 text-white rounded-full py-2 text-sm hover:bg-blue-700">
              Try for free
            </button>
          </div>
        </nav>
      </div>
    </header>
  );
}
