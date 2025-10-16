import { Link } from "react-router-dom";
import { Target, Users, Lightbulb, ArrowRight, Sparkles, Globe, Zap, Heart, TrendingUp, Shield, CheckCircle2 } from "lucide-react";
import homeImage from "../assets/img/home1.png";

export function AboutUsPage() {
  return (
    <div
      className="min-h-screen bg-cover bg-no-repeat relative"
      style={{ backgroundImage: `url(${homeImage})` }}
    >
      {/* Animated Background Overlay */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute top-20 left-10 w-96 h-96 bg-green-400/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-[500px] h-[500px] bg-blue-400/15 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-purple-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 lg:px-8 pt-24 pb-20">
        <div className="text-center">
          {/* <div className="inline-flex items-center gap-2 px-5 py-2 mb-6 bg-gradient-to-r from-green-500 to-blue-500 rounded-full shadow-lg animate-pulse">
            <Sparkles className="w-5 h-5 text-white" />
            <span className="text-sm font-bold text-white">Transforming Agriculture with AI</span>
          </div> */}

          <h1 className="text-5xl md:text-7xl font-extrabold text-gray-900 mb-6 leading-tight">
            About <span className="bg-gradient-to-r from-green-600 via-emerald-600 to-blue-600 bg-clip-text text-transparent">AgriFinSight</span>
          </h1>
          <p className="text-2xl text-gray-700 mb-8 max-w-4xl mx-auto leading-relaxed">
            Empowering farmers with AI-driven insights to revolutionize
            agriculture and maximize crop yields across Africa and beyond.
          </p>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto mt-12">
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-lg border border-green-200">
              <div className="text-4xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent mb-2">50K+</div>
              <div className="text-sm font-semibold text-gray-700">Active Farmers</div>
            </div>
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-lg border border-blue-200">
              <div className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">98%</div>
              <div className="text-sm font-semibold text-gray-700">Detection Accuracy</div>
            </div>
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-lg border border-purple-200">
              <div className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">35%</div>
              <div className="text-sm font-semibold text-gray-700">Yield Increase</div>
            </div>
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-lg border border-amber-200">
              <div className="text-4xl font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent mb-2">24/7</div>
              <div className="text-sm font-semibold text-gray-700">AI Monitoring</div>
            </div>
          </div>
        </div>
      </section>

      {/* Mission & Vision Section */}
      <section className="bg-gradient-to-b from-white/90 to-green-50/80 py-20">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-block mb-4">
              <div className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-green-100 to-blue-100 rounded-full border-2 border-green-200">
                <Heart className="w-5 h-5 text-green-600" />
                <span className="text-sm font-bold text-green-700">Our Purpose</span>
              </div>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              Driven by <span className="bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">passion</span> and <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">purpose</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-2 gap-12">
            {/* Our Mission */}
            <div className="group relative bg-white shadow-soft-2xl rounded-3xl p-10 border-2 border-gray-100 hover:border-green-300 transition-all duration-300 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-green-50 to-emerald-50 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative">
                <div className="w-20 h-20 bg-gradient-to-br from-green-600 to-lime-400 rounded-2xl flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform">
                  <Target className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-6">
                  Our Mission
                </h2>
                <p className="text-gray-600 text-lg leading-relaxed mb-6">
                  To provide accessible, AI-powered agricultural solutions that
                  help farmers detect crop diseases early, optimize farming
                  practices, and increase productivity through data-driven
                  insights. We believe technology can bridge the gap between
                  traditional farming knowledge and modern agricultural science.
                </p>
                <div className="flex items-center gap-2 text-green-600 font-semibold">
                  <CheckCircle2 className="w-5 h-5" />
                  <span>Empowering farmers worldwide</span>
                </div>
              </div>
            </div>

            {/* Our Vision */}
            <div className="group relative bg-white shadow-soft-2xl rounded-3xl p-10 border-2 border-gray-100 hover:border-blue-300 transition-all duration-300 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-cyan-400 rounded-2xl flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform">
                  <Globe className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-bold text-gray-900 mb-6">
                  Our Vision
                </h2>
                <p className="text-gray-600 text-lg leading-relaxed mb-6">
                  To become the leading agricultural intelligence platform in
                  Africa, transforming farming communities through innovative
                  technology and sustainable practices. We envision a future where
                  every farmer has access to cutting-edge tools that enhance their
                  decision-making and improve food security.
                </p>
                <div className="flex items-center gap-2 text-blue-600 font-semibold">
                  <CheckCircle2 className="w-5 h-5" />
                  <span>Leading Africa's agritech revolution</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* What We Do Section */}
      <section className="py-20 bg-white/80">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-block mb-4">
              <div className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full border-2 border-purple-200">
                <Zap className="w-5 h-5 text-purple-600" />
                <span className="text-sm font-bold text-purple-700">Our Solutions</span>
              </div>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              What We <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">Do</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              AgriFinSight combines advanced machine learning with agricultural
              expertise to deliver powerful solutions for modern farmers.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "AI Disease Detection",
                desc: "Upload images of your crops to instantly identify diseases, pests, and nutrient deficiencies with over 98% accuracy.",
                color: "from-purple-600 to-pink-500",
                icon: Target,
                emoji: "🔬"
              },
              {
                title: "Smart Analytics",
                desc: "Get detailed crop health reports, growth predictions, and yield forecasts based on historical data and weather patterns.",
                color: "from-amber-500 to-orange-500",
                icon: TrendingUp,
                emoji: "📊"
              },
              {
                title: "Personalized Recommendations",
                desc: "Receive tailored advice on treatments, fertilizers, irrigation, and planting schedules optimized for your specific farm conditions.",
                color: "from-green-500 to-lime-500",
                icon: Lightbulb,
                emoji: "💡"
              },
            ].map((item, i) => {
              const Icon = item.icon
              return (
                <div
                  key={i}
                  className="group relative bg-white shadow-xl rounded-3xl p-8 hover:shadow-2xl transition-all duration-300 hover:scale-105 border-2 border-gray-100 hover:border-transparent"
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${item.color} opacity-0 group-hover:opacity-5 rounded-3xl transition-opacity`}></div>
                  <div className="relative">
                    <div className="flex items-center justify-between mb-6">
                      <div
                        className={`w-16 h-16 bg-gradient-to-br ${item.color} rounded-2xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform`}
                      >
                        <Icon className="w-8 h-8 text-white" />
                      </div>
                      <div className="text-5xl">{item.emoji}</div>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-4">
                      {item.title}
                    </h3>
                    <p className="text-gray-600 leading-relaxed text-lg mb-6">{item.desc}</p>
                    <div className="flex items-center gap-2 text-gray-700 font-semibold">
                      <CheckCircle2 className={`w-5 h-5 bg-gradient-to-r ${item.color} text-white rounded-full p-0.5`} />
                      <span className="text-sm">Trusted by 50K+ farmers</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Our Team/Values Section */}
      <section className="bg-gradient-to-b from-white/90 to-blue-50/80 py-20">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-block mb-4">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 rounded-3xl flex items-center justify-center mx-auto shadow-2xl animate-pulse">
                <Users className="w-12 h-12 text-white" />
              </div>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6 mt-6">
              Our Core <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Values</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              We are driven by a commitment to excellence, innovation, and
              sustainable agricultural practices that benefit farmers worldwide.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                title: "Innovation",
                desc: "Continuously improving our AI models and technology stack to stay ahead.",
                icon: Zap,
                gradient: "from-yellow-500 to-orange-500",
                bgGradient: "from-yellow-50 to-orange-50"
              },
              {
                title: "Accessibility",
                desc: "Making advanced agricultural tools available to all farmers, everywhere.",
                icon: Users,
                gradient: "from-green-500 to-emerald-500",
                bgGradient: "from-green-50 to-emerald-50"
              },
              {
                title: "Sustainability",
                desc: "Promoting eco-friendly farming practices and resource conservation.",
                icon: Globe,
                gradient: "from-blue-500 to-cyan-500",
                bgGradient: "from-blue-50 to-cyan-50"
              },
              {
                title: "Accuracy",
                desc: "Delivering reliable, science-backed insights farmers can trust completely.",
                icon: Shield,
                gradient: "from-purple-500 to-pink-500",
                bgGradient: "from-purple-50 to-pink-50"
              },
            ].map((value, i) => {
              const Icon = value.icon
              return (
                <div
                  key={i}
                  className="group relative bg-white shadow-lg rounded-2xl p-8 text-center hover:shadow-2xl transition-all duration-300 hover:scale-105 border-2 border-gray-100 hover:border-transparent"
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${value.bgGradient} rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity`}></div>
                  <div className="relative">
                    <div className={`w-16 h-16 bg-gradient-to-br ${value.gradient} rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg group-hover:scale-110 group-hover:rotate-6 transition-all`}>
                      <Icon className="w-8 h-8 text-white" />
                    </div>
                    <h4 className="text-xl font-bold text-gray-900 mb-3">
                      {value.title}
                    </h4>
                    <p className="text-gray-600 leading-relaxed">{value.desc}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative bg-gradient-to-br from-green-600 via-emerald-600 to-teal-600 py-24 overflow-hidden">
        <div className="absolute top-0 left-0 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-white/3 rounded-full blur-3xl"></div>

        <div className="relative max-w-7xl mx-auto px-6 lg:px-8 text-center text-white">
          <div className="inline-flex items-center gap-2 px-5 py-2 mb-6 bg-white/20 backdrop-blur-sm rounded-full border-2 border-white/30 shadow-lg">
            <Sparkles className="w-5 h-5 text-white animate-pulse" />
            <span className="text-sm font-bold text-white">Join 50K+ Farmers Today</span>
          </div>

          <h2 className="text-4xl md:text-6xl font-extrabold mb-6">
            Join Our Growing Community
          </h2>
          <p className="text-xl md:text-2xl text-white/90 mb-12 max-w-3xl mx-auto leading-relaxed">
            Be part of the agricultural revolution. Start using AgriFinSight
            today and experience the power of AI-driven farming insights that increase yields by 35%.
          </p>

          {/* Feature Pills */}
          <div className="flex flex-wrap gap-4 justify-center mb-12">
            <div className="flex items-center gap-2 px-5 py-3 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 shadow-lg">
              <CheckCircle2 className="w-5 h-5 text-white" />
              <span className="text-sm font-semibold">Free to Start</span>
            </div>
            <div className="flex items-center gap-2 px-5 py-3 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 shadow-lg">
              <CheckCircle2 className="w-5 h-5 text-white" />
              <span className="text-sm font-semibold">No Credit Card</span>
            </div>
            <div className="flex items-center gap-2 px-5 py-3 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 shadow-lg">
              <CheckCircle2 className="w-5 h-5 text-white" />
              <span className="text-sm font-semibold">Expert Support</span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-6 justify-center">
            <Link
              to="/register"
              className="group bg-white text-green-700 hover:bg-gray-50 px-10 py-5 text-lg font-bold rounded-2xl inline-flex items-center justify-center transition-all shadow-2xl hover:shadow-3xl hover:scale-110"
            >
              Get Started Free
              <ArrowRight className="ml-2 w-6 h-6 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/ai-features"
              className="group bg-white/10 backdrop-blur-sm border-2 border-white/30 text-white hover:bg-white/20 px-10 py-5 text-lg font-bold rounded-2xl inline-flex items-center justify-center transition-all hover:scale-110"
            >
              Explore Features
              <Sparkles className="ml-2 w-5 h-5 group-hover:rotate-12 transition-transform" />
            </Link>
          </div>

          {/* Social Proof */}
          <div className="mt-16 pt-10 border-t border-white/20">
            <p className="text-white/70 text-sm mb-6">Trusted by farmers across Africa and beyond</p>
            <div className="flex flex-wrap items-center justify-center gap-6 text-white/60">
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">50K+</div>
                <div className="text-sm">Active Users</div>
              </div>
              <div className="hidden sm:block w-px h-12 bg-white/20"></div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">98%</div>
                <div className="text-sm">Accuracy Rate</div>
              </div>
              <div className="hidden sm:block w-px h-12 bg-white/20"></div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-1">24/7</div>
                <div className="text-sm">Support</div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
