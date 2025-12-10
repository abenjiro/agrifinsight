import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Eye, EyeOff, Mail, Lock, AlertCircle, CheckCircle, X } from "lucide-react";
import { authService } from '../services/api';

export function LoginPage() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");
  const navigate = useNavigate();

  // Redirect if already logged in
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      navigate('/dashboard');
    }
  }, [navigate]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    e.stopPropagation();

    setError("");
    setSuccess("");
    setLoading(true);

    try {
      const response : any = await authService.login({
        email: formData.email,
        password: formData.password,
      });

      // Store the access token
      localStorage.setItem('auth_token', response.access_token);

      // Store user data
      localStorage.setItem('user', JSON.stringify(response.user));

      // Show success message
      setSuccess('Welcome back! Redirecting to dashboard...');

      // Navigate after a brief delay to show the success message
      setTimeout(() => {
        navigate('/dashboard');
      }, 1000);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.response?.detail || error.message || 'Login failed. Please check your credentials.';
      setError(errorMessage);
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 
      bg-gradient-to-br from-green-100 via-amber-100 to-emerald-200"
    >
      <div className="max-w-md w-full space-y-4 bg-white/70 backdrop-blur-md shadow-xl rounded-2xl p-8 overflow-visible">
        {/* Logo / Header */}
        <div className="text-center">
          <div className="flex justify-center">
            <div className="w-12 h-12 bg-green-600 rounded-xl flex items-center justify-center shadow-md">
              <span className="text-white font-bold text-xl">A</span>
            </div>
          </div>
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            Welcome Back
          </h2>
          <p className="mt-2 text-sm text-gray-600">
            Sign in to continue or{" "}
            <Link
              to="/register"
              className="font-medium text-green-700 hover:text-green-600 transition"
            >
              create an account
            </Link>
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-sm font-medium text-red-800">Login Failed</h3>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
            <button
              onClick={() => setError("")}
              className="text-red-400 hover:text-red-600 transition-colors"
              aria-label="Close error message"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {/* Success Message */}
        {success && (
          <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg flex items-start gap-3">
            <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-medium text-green-800">Login Successful</h3>
              <p className="mt-1 text-sm text-green-700">{success}</p>
            </div>
          </div>
        )}

        {/* Form */}
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-5">
            {/* Email */}
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700"
              >
                Email address
              </label>
              <div className="mt-1 relative">
                <Mail className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="input pl-10 w-full border-gray-300 rounded-lg focus:ring-green-500 focus:border-green-500"
                  placeholder="you@example.com"
                  value={formData.email}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                Password
              </label>
              <div className="mt-1 relative">
                <Lock className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  required
                  className="input pl-10 pr-10 w-full border-gray-300 rounded-lg focus:ring-green-500 focus:border-green-500"
                  placeholder="••••••••"
                  value={formData.password}
                  onChange={handleChange}
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-3 flex items-center text-gray-400 hover:text-gray-600"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeOff className="h-5 w-5" />
                  ) : (
                    <Eye className="h-5 w-5" />
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Options */}
          <div className="flex items-center justify-between text-sm">
            <label className="flex items-center">
              <input
                type="checkbox"
                className="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
              />
              <span className="ml-2 text-gray-700">Remember me</span>
            </label>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                // TODO: Implement forgot password
              }}
              className="text-green-700 hover:text-green-600"
            >
              Forgot password?
            </button>
          </div>

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 mt-4 rounded-lg bg-green-600 text-white font-semibold hover:bg-green-700 transition-all shadow-md"
            >
              {loading ? "Signing in..." : "Sign In"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

