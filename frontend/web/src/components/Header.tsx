import { Home, Inbox, BarChart3, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

const navigationLinks = [
  { label: "Home", href: "/", icon: Home, active: true },
  { label: "Features", href: "#features", icon: Inbox, active: false },
  { label: "About Us", href: "#about", icon: BarChart3, active: false },
];

const Logo = () => (
//   <div className="animate-spin rounded-full border-2 border-gray-400 border-t-transparent size-6" />
  <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">A</span>
    </div>
);


export default function Navbar() {
  return (
    <header className="border-b px-4 md:px-6">
      <div className="flex h-16 items-center justify-between gap-4">
        {/* Left Nav */}
        <div className="flex flex-1 items-center gap-6">
          {navigationLinks.map((link, idx) => {
            const Icon = link.icon;
            return (
              <Link
                key={idx}
                to={link.href}
                className={`flex items-center gap-2 text-sm font-medium ${
                  link.active ? "text-black font-semibold" : "text-gray-600 hover:text-black"
                }`}
              >
                <Icon size={16} className="opacity-70" />
                {link.label}
              </Link>
            );
          })}
        </div>

        {/* Middle: Logo */}
        <div className="flex items-center">
          <Logo />
        </div>

        {/* Right Side */}
        <div className="flex flex-1 items-center justify-end gap-4">
          <Link 
            to="/login" 
            className="text-gray-600 hover:text-black transition-colors text-sm font-medium"
          >
            Sign In
          </Link>
          <Link 
            to="/register" 
            className="flex items-center gap-1 rounded-lg bg-black px-3 py-1 text-white text-sm hover:bg-gray-800 transition-colors"
          >
            <Sparkles size={16} className="opacity-70" />
            <span className="hidden sm:inline">Get Started</span>
          </Link>
        </div>
      </div>
    </header>
  );
}
