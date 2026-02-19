import { motion } from "framer-motion";
import { 
  MessageSquare, 
  Settings, 
  UserCircle, 
  Mic
} from "lucide-react";
import { cn } from "@/lib/utils";
import { MimicLogo } from "./MimicLogo";

interface SidebarProps {
  activeView: "chat" | "settings" | "personas" | "voice";
  onViewChange: (view: "chat" | "settings" | "personas" | "voice") => void;
}

const navItems = [
  { id: "chat" as const, label: "Chat", icon: MessageSquare },
  { id: "personas" as const, label: "Personas", icon: UserCircle },
  { id: "voice" as const, label: "Voice", icon: Mic },
  { id: "settings" as const, label: "Settings", icon: Settings },
];

export function Sidebar({ activeView, onViewChange }: SidebarProps) {
  return (
    <motion.aside
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      className="w-20 bg-card border-r border-border flex flex-col items-center py-6 gap-2"
    >
      {/* Logo */}
      <div className="mb-8">
        <motion.div 
          className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center overflow-hidden"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <MimicLogo size={36} className="drop-shadow-md" />
        </motion.div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-col gap-2 flex-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={cn(
                "relative w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 group",
                isActive 
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/25" 
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              )}
            >
              <Icon className="w-5 h-5" />
              
              {/* Tooltip */}
              <span className="absolute left-full ml-3 px-2 py-1 bg-popover text-popover-foreground text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50">
                {item.label}
              </span>
              
              {/* Active indicator */}
              {isActive && (
                <motion.div
                  layoutId="activeNav"
                  className="absolute inset-0 rounded-xl bg-primary -z-10"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                />
              )}
            </button>
          );
        })}
      </nav>

      {/* Status indicator */}
      <div className="mt-auto flex flex-col items-center gap-2">
        <motion.div 
          className="w-3 h-3 rounded-full bg-green-500"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
          title="System Online"
        />
      </div>
    </motion.aside>
  );
}
