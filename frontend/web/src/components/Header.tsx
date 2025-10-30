import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Menu, X } from "lucide-react";

export default function Header() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navItems = [
    { label: "Home", href: "/", active: true },
    { label: "AI Features", href: "/ai-features" },
    { label: "About Us", href: "/about" },
    { label: "Contact", href: "/contact" },
  ];

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
            <Link
              key={item.label}
              to={item.href}
              className={`transition hover:opacity-80 ${
                item.active ? "font-semibold" : ""
              }`}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Desktop Buttons */}
        <div className="hidden md:flex gap-3">
          <Link
            to="/register"
            className={`px-4 py-2 rounded-full border text-sm font-medium transition
            ${
              scrolled
                ? "border-green-300 hover:bg-green-50 text-green-700"
                : "border-white/30 hover:bg-white/10 text-white"
            }`}
          >
            Sign Up
          </Link>
          <Link
            to="/login"
            className={`px-4 py-2 rounded-full text-sm font-semibold transition
            ${
              scrolled
                ? "bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-700 hover:to-emerald-700"
                : "bg-white text-green-700 hover:bg-green-50"
            }`}
          >
            Sign In
          </Link>
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
            <Link
              key={item.label}
              to={item.href}
              className="text-sm font-medium"
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </Link>
          ))}

          {/* Mobile Buttons */}
          <div className="border-t pt-4 mt-4 flex flex-col gap-3">
            <Link
              to="/register"
              className="w-full border border-green-300 text-green-700 rounded-full py-2 text-sm hover:bg-green-50 text-center font-medium"
              onClick={() => setMenuOpen(false)}
            >
              Sign Up
            </Link>
            <Link
              to="/login"
              className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-full py-2 text-sm hover:from-green-700 hover:to-emerald-700 text-center font-semibold"
              onClick={() => setMenuOpen(false)}
            >
              Sign In
            </Link>
          </div>
        </nav>
      </div>
    </header>
  );
}
