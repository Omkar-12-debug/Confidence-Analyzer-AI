import { NavLink, useLocation } from "react-router-dom";
import { Home, BarChart3, Clock, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import PageTransition from "./PageTransition";

const navItems = [
  { title: "Home", path: "/", icon: Home },
  { title: "Analysis", path: "/analysis", icon: BarChart3 },
  { title: "History", path: "/history", icon: Clock },
  { title: "About", path: "/about", icon: Info },
];

const Layout = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-50 border-b bg-card/80 backdrop-blur-md">
        <div className="container flex h-16 items-center justify-between">
          <motion.div
            className="flex items-center gap-2"
            whileHover={{ scale: 1.03 }}
            transition={{ type: "spring", stiffness: 400, damping: 15 }}
          >
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
              <BarChart3 className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="font-heading font-bold text-lg">StressAI</span>
          </motion.div>
          <nav className="flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200",
                    isActive
                      ? "text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  )}
                >
                  {isActive && (
                    <motion.div
                      layoutId="nav-active"
                      className="absolute inset-0 bg-primary rounded-lg shadow-sm"
                      transition={{ type: "spring", stiffness: 380, damping: 28 }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-2">
                    <item.icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{item.title}</span>
                  </span>
                </NavLink>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="flex-1">
        <PageTransition>{children}</PageTransition>
      </main>
    </div>
  );
};

export default Layout;
