

Try something like this, in your user key binding:

    {
        "keys": ["YOUR_SHORTCUT"],
        "command": "set_setting",
        "args":
        {
            "setting": "color_scheme",
            "value": "Packages/Color Scheme - Default/Solarized (Light).tmTheme"
        }
    }

Of course, change `Packages/Color Scheme - Default/Solarized (Light).tmTheme` to whatever theme you prefer.

If you want a toggle between two color schemes, you can create a plugin (`Tools/New Plugin...`):

    import sublime, sublime_plugin

    class ToggleColorSchemeCommand(sublime_plugin.TextCommand):
        def run(self, edit, **args):

            scheme1 = args["color_scheme_1"]
            scheme2 = args["color_scheme_2"]
            current_scheme = self.view.settings().get("color_scheme")

            new_scheme = scheme1 if current_scheme == scheme2 else scheme2
            self.view.settings().set("color_scheme", new_scheme)

and save it in your `Packages/User` directory.

Then add a key binding like this:

    {
        "keys": ["YOUR_TOGGLE_SHORCUT"], "command": "toggle_color_scheme",
        "args":
        {
            "color_scheme_1": "Packages/Color Scheme - Default/Solarized (Light).tmTheme" ,
            "color_scheme_2": "Packages/Color Scheme - Default/Solarized (Dark).tmTheme"
        }
    }

---

**CREDIT**: [riccardo-marotti](https://stackoverflow.com/users/761777/riccardo-marotti)

**URL**: https://stackoverflow.com/a/13121974
