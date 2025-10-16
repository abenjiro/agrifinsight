import { Link } from 'react-router-dom'
import {
  Sparkles,
  Brain,
  Target,
  TrendingUp,
  Satellite,
  AlertTriangle,
  CloudRain,
  Leaf,
  CheckCircle2,
  ArrowRight,
  Zap,
  Shield,
  BarChart3
} from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'AI Crop Disease Detection',
    description: 'Upload a photo of your crop and get instant disease identification powered by advanced deep learning models.',
    benefits: [
      '98% accuracy in disease detection',
      'Identify 50+ crop diseases instantly',
      'Get treatment recommendations',
      'Track disease progression over time'
    ],
    image: '🌾',
    gradient: 'from-red-500 to-orange-500',
    bgGradient: 'from-red-50 to-orange-50'
  },
  {
    icon: Target,
    title: 'Smart Planting Recommendations',
    description: 'AI analyzes your soil, climate, and market trends to recommend the best crops and optimal planting times.',
    benefits: [
      'Maximize yield potential by 35%',
      'Reduce crop failure risk',
      'Weather-based planting calendar',
      'Market price predictions'
    ],
    image: '🌱',
    gradient: 'from-green-500 to-emerald-500',
    bgGradient: 'from-green-50 to-emerald-50'
  },
  {
    icon: TrendingUp,
    title: 'Harvest Yield Predictions',
    description: 'Machine learning models forecast your harvest yield and quality weeks in advance for better planning.',
    benefits: [
      'Predict yields with 92% accuracy',
      'Plan harvest logistics in advance',
      'Optimize resource allocation',
      'Maximize profit margins'
    ],
    image: '📊',
    gradient: 'from-blue-500 to-cyan-500',
    bgGradient: 'from-blue-50 to-cyan-50'
  },
  {
    icon: Satellite,
    title: 'Satellite Crop Monitoring',
    description: 'Track crop health from space using NDVI analysis and satellite imagery for large-scale farm monitoring.',
    benefits: [
      'Monitor farms from anywhere',
      'Detect stress before visible',
      'Track growth patterns',
      'Historical land use insights'
    ],
    image: '🛰️',
    gradient: 'from-purple-500 to-pink-500',
    bgGradient: 'from-purple-50 to-pink-50'
  },
  {
    icon: CloudRain,
    title: 'Weather Intelligence',
    description: 'AI-powered weather forecasting and alerts help you make informed decisions about irrigation and protection.',
    benefits: [
      '7-day hyperlocal forecasts',
      'Rainfall predictions',
      'Frost and heat alerts',
      'Irrigation scheduling'
    ],
    image: '🌦️',
    gradient: 'from-indigo-500 to-blue-500',
    bgGradient: 'from-indigo-50 to-blue-50'
  },
  {
    icon: AlertTriangle,
    title: 'Pest & Disease Alerts',
    description: 'Early warning system uses AI to predict pest outbreaks and disease spread in your region.',
    benefits: [
      'Prevent outbreaks proactively',
      'Region-based risk alerts',
      'Treatment timing optimization',
      'Reduce pesticide costs by 40%'
    ],
    image: '🐛',
    gradient: 'from-yellow-500 to-amber-500',
    bgGradient: 'from-yellow-50 to-amber-50'
  }
]

const stats = [
  { value: '98%', label: 'Disease Detection Accuracy', icon: Target },
  { value: '35%', label: 'Average Yield Increase', icon: TrendingUp },
  { value: '50K+', label: 'Farms Optimized', icon: Leaf },
  { value: '24/7', label: 'AI Monitoring', icon: Zap }
]

export function AIFeaturesPage() {
  return (
    <div className="w-full min-h-screen">
      {/* Animated Background */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-blue-100 to-purple-50"></div>
        <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-green-400/30 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-400/35 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-40 left-1/3 w-[400px] h-[400px] bg-purple-400/25 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-cyan-400/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1.5s' }}></div>
      </div>

      {/* Hero Section */}
      <div className="relative overflow-hidden pt-32 pb-12">
        <div className="absolute inset-0 bg-gradient-to-br from-green-400/10 via-blue-400/10 to-purple-400/5"></div>
        <div className="absolute top-0 left-0 right-0 h-96 bg-gradient-to-b from-green-200/20 via-blue-200/15 to-transparent"></div>
        <div className="relative max-w-7xl mx-auto px-6 mb-12">
          <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border overflow-hidden">
            <div className="p-8 md:p-12">
              <div className="flex flex-col lg:flex-row items-center gap-8">
                <div className="flex-1">
                  <div className="inline-flex items-center gap-2 px-4 py-2 mb-4 bg-gradient-to-r from-green-400 to-blue-500 rounded-full shadow-lg animate-pulse">
                    <Sparkles className="w-4 h-4 text-white" />
                    <span className="text-sm font-semibold text-white">AI-Powered Agriculture</span>
                  </div>

                  <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4 leading-tight">
                    Transform Your Farm with
                    <span className="bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent"> Artificial Intelligence</span>
                  </h1>

                  <p className="text-lg text-gray-600 mb-6 leading-relaxed">
                    Harness the power of cutting-edge AI and satellite technology to maximize yields,
                    prevent crop diseases, and make data-driven decisions that boost your profits.
                  </p>

                  <div className="flex flex-wrap gap-4">
                    <Link
                      to="/register"
                      className="inline-flex items-center px-8 py-3.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-sm ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
                    >
                      Get Started Free
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Link>

                    <Link
                      to="/analysis"
                      className="inline-flex items-center px-8 py-3.5 font-bold text-center text-gray-700 uppercase align-middle transition-all bg-white border border-solid rounded-lg cursor-pointer border-gray-300 leading-pro text-sm ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
                    >
                      Try Demo
                    </Link>
                  </div>
                </div>

                {/* Hero Visual */}
                <div className="flex-1 relative">
                  <div className="relative">
                    <div className="absolute -top-4 -right-4 w-72 h-72 bg-gradient-to-br from-green-400 to-blue-500 rounded-full blur-3xl opacity-30 animate-pulse"></div>
                    <div className="relative bg-gradient-to-br from-green-100 via-blue-100 to-purple-100 rounded-2xl p-8 border-2 border-green-200 shadow-xl">
                      <div className="text-8xl mb-4 text-center">🤖🌾</div>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-white rounded-lg p-3 shadow-sm">
                          <div className="flex items-center gap-2">
                            <CheckCircle2 className="w-4 h-4 text-green-600" />
                            <span className="text-xs font-semibold">Disease Detected</span>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">98% confidence</p>
                        </div>
                        <div className="bg-white rounded-lg p-3 shadow-sm">
                          <div className="flex items-center gap-2">
                            <TrendingUp className="w-4 h-4 text-blue-600" />
                            <span className="text-xs font-semibold">Yield +35%</span>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">vs last season</p>
                        </div>
                        <div className="bg-white rounded-lg p-3 shadow-sm">
                          <div className="flex items-center gap-2">
                            <Satellite className="w-4 h-4 text-purple-600" />
                            <span className="text-xs font-semibold">NDVI: 0.82</span>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">Healthy crops</p>
                        </div>
                        <div className="bg-white rounded-lg p-3 shadow-sm">
                          <div className="flex items-center gap-2">
                            <CloudRain className="w-4 h-4 text-indigo-600" />
                            <span className="text-xs font-semibold">Rain Alert</span>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">in 2 hours</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="max-w-7xl mx-auto px-6 mb-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((stat, index) => {
            const Icon = stat.icon
            return (
              <div
                key={index}
                className="relative flex flex-col min-w-0 break-words bg-white border-2 border-green-100 shadow-soft-xl rounded-2xl bg-clip-border hover:shadow-soft-2xl hover:border-green-300 hover:scale-105 transition-all duration-300"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-green-50/50 to-blue-50/50 rounded-2xl opacity-0 hover:opacity-100 transition-opacity"></div>
                <div className="relative p-6 text-center">
                  <div className="inline-flex items-center justify-center w-12 h-12 mb-3 bg-gradient-to-br from-green-500 to-blue-600 rounded-xl shadow-lg">
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">{stat.value}</h3>
                  <p className="text-sm text-gray-600 mt-1 font-medium">{stat.label}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Features Grid */}
      <div className="max-w-7xl mx-auto px-6 mb-12">
        <div className="text-center mb-8">
          <div className="inline-block mb-4">
            <div className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full border-2 border-purple-200">
              <Sparkles className="w-5 h-5 text-purple-600" />
              <span className="text-sm font-bold text-purple-700">6 Powerful AI Tools</span>
            </div>
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-3">
            Powerful AI Features for Modern Farmers
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Everything you need to optimize your farm operations and maximize profitability
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div
                key={index}
                className="relative flex flex-col min-w-0 break-words bg-white border-2 border-gray-100 shadow-soft-xl rounded-2xl bg-clip-border hover:shadow-soft-2xl hover:border-transparent transition-all duration-300 group overflow-hidden"
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.bgGradient} rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300`}></div>
                <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br ${feature.gradient} opacity-5 rounded-full -mr-16 -mt-16`}></div>

                <div className="relative p-6">
                  {/* Icon & Title */}
                  <div className="flex items-start gap-4 mb-4">
                    <div className={`flex-shrink-0 w-14 h-14 flex items-center justify-center rounded-xl bg-gradient-to-br ${feature.gradient} shadow-lg`}>
                      <Icon className="w-7 h-7 text-white" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold text-gray-900 mb-2">{feature.title}</h3>
                      <p className="text-sm text-gray-600 leading-relaxed">{feature.description}</p>
                    </div>
                    <div className="text-4xl">{feature.image}</div>
                  </div>

                  {/* Benefits List */}
                  <div className="space-y-2 mt-4">
                    {feature.benefits.map((benefit, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <CheckCircle2 className={`w-5 h-5 flex-shrink-0 mt-0.5 bg-gradient-to-br ${feature.gradient} text-white rounded-full p-0.5`} />
                        <span className="text-sm text-gray-700">{benefit}</span>
                      </div>
                    ))}
                  </div>

                  {/* CTA */}
                  <div className="mt-6 pt-4 border-t border-gray-100">
                    <Link
                      to="/register"
                      className={`inline-flex items-center text-sm font-semibold bg-gradient-to-r ${feature.gradient} bg-clip-text text-transparent group-hover:translate-x-1 transition-transform`}
                    >
                      Start using this feature
                      <ArrowRight className="w-4 h-4 ml-1" />
                    </Link>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Trust Badges */}
      <div className="max-w-7xl mx-auto px-6 mb-12">
        <div className="relative flex flex-col min-w-0 break-words bg-gradient-to-br from-gray-900 via-gray-800 to-green-900 border-0 shadow-soft-xl rounded-2xl bg-clip-border overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-green-500/10 rounded-full blur-3xl"></div>
          <div className="relative p-8 text-center">
            <div className="inline-block mb-3">
              <div className="flex items-center gap-2 px-5 py-2 bg-white/10 backdrop-blur-sm rounded-full border-2 border-white/20">
                <Shield className="w-5 h-5 text-green-400" />
                <span className="text-sm font-bold text-white">Trusted by 50K+ Farmers</span>
              </div>
            </div>
            <h3 className="text-2xl font-bold text-white mb-6">Why Farmers Trust Our AI</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 flex items-center justify-center bg-white/10 rounded-full mb-3">
                  <Shield className="w-8 h-8 text-green-400" />
                </div>
                <h4 className="text-white font-semibold mb-2">Data Privacy</h4>
                <p className="text-gray-400 text-sm">Your farm data is encrypted and secure</p>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 flex items-center justify-center bg-white/10 rounded-full mb-3">
                  <Brain className="w-8 h-8 text-blue-400" />
                </div>
                <h4 className="text-white font-semibold mb-2">Proven AI Models</h4>
                <p className="text-gray-400 text-sm">Trained on millions of crop images</p>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 flex items-center justify-center bg-white/10 rounded-full mb-3">
                  <BarChart3 className="w-8 h-8 text-purple-400" />
                </div>
                <h4 className="text-white font-semibold mb-2">Real Results</h4>
                <p className="text-gray-400 text-sm">35% average yield improvement</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Final CTA */}
      <div className="max-w-7xl mx-auto px-6 mb-12 pb-12">
        <div className="relative flex flex-col min-w-0 break-words bg-gradient-to-br from-green-600 via-blue-600 to-purple-600 border-0 shadow-soft-xl rounded-2xl bg-clip-border overflow-hidden">
          <div className="absolute top-0 left-0 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
          <div className="relative p-12 text-center">
            <div className="text-6xl mb-4 animate-bounce">🚀</div>
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Transform Your Farm?
            </h2>
            <p className="text-lg text-white/90 mb-8 max-w-2xl mx-auto">
              Join thousands of farmers who are already using AI to increase yields and reduce costs.
              Get started today with a free account!
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                to="/register"
                className="inline-flex items-center px-8 py-3.5 font-bold text-center text-green-600 uppercase align-middle transition-all bg-white rounded-lg cursor-pointer leading-pro text-sm ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
              >
                Create Free Account
                <ArrowRight className="w-4 h-4 ml-2" />
              </Link>
              <Link
                to="/dashboard"
                className="inline-flex items-center px-8 py-3.5 font-bold text-center text-white uppercase align-middle transition-all bg-white/10 backdrop-blur border-2 border-white/30 rounded-lg cursor-pointer leading-pro text-sm ease-soft-in tracking-tight-soft hover:bg-white/20"
              >
                View Dashboard
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
