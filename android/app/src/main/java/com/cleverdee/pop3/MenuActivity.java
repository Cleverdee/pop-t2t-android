package com.cleverdee.pop3;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/* The start screen: play, or manage the game files. If the game crashed last time, a crash-report screen
   comes first. */
public class MenuActivity extends Activity {
    private static final String GAME = "Prince of Persia: The Two Thrones";

    private LinearLayout mRoot;
    private boolean mMain;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mRoot = Ui.page(this);
        File crash = new File(getExternalFilesDir(null), "crash.txt");
        if (crash.isFile()) showCrash(crash, this::showMain);
        else showMain();
    }

    /* The game crashed last time (host/window_android.c or GameActivity wrote crash.txt): offer to share a report,
       the crash and the end of the game's log, through any app the player likes. */
    private void showCrash(File crash, Runnable next) {
        mMain = false;
        Ui.clear(mRoot);
        Ui.heading(mRoot, "The game closed unexpectedly");
        text("Help get it fixed: share or copy the crash report and send it to the developer any way you like, "
           + "on Reddit, GitHub or in a message. It has your phone model, Android version and the game's log; "
           + "nothing personal.", 15, Ui.SOFT);
        final String report = crashReport(crash);
        crash.delete();
        button("Share report", v -> {
            Intent send = new Intent(Intent.ACTION_SEND);
            send.setType("text/plain");
            send.putExtra(Intent.EXTRA_SUBJECT, GAME + " crash report");
            send.putExtra(Intent.EXTRA_TEXT, report);
            try { startActivity(Intent.createChooser(send, "Send the crash report")); }
            catch (android.content.ActivityNotFoundException e) { /* nothing to send it with */ }
        });
        button("Copy report", v -> {
            android.content.ClipboardManager cb = (android.content.ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            cb.setPrimaryClip(android.content.ClipData.newPlainText(GAME + " crash report", report));
            android.widget.Toast.makeText(this, "Copied", android.widget.Toast.LENGTH_SHORT).show();
        });
        Ui.button(mRoot, "Continue", true, v -> next.run());
    }

    private String crashReport(File crash) {
        StringBuilder r = new StringBuilder(GAME + " crash report\n");
        String version = "?";
        try { version = getPackageManager().getPackageInfo(getPackageName(), 0).versionName; } catch (Exception e) { /* unknown */ }
        r.append("App ").append(version).append(", ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
         .append(", Android ").append(Build.VERSION.RELEASE).append(" (API ").append(Build.VERSION.SDK_INT).append(')');
        if (Build.VERSION.SDK_INT >= 31) r.append(", ").append(Build.SOC_MANUFACTURER).append(' ').append(Build.SOC_MODEL);
        r.append("\n\n");
        for (String l : readLines(crash)) r.append(l).append('\n');
        List<String> log = readLines(new File(getExternalFilesDir(null), "last_log.txt"));
        r.append("\nlog (last lines):\n");
        for (int i = Math.max(0, log.size() - 120); i < log.size(); i++) r.append(log.get(i)).append('\n');
        return r.toString();
    }

    static List<String> readLines(File f) {
        List<String> out = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(f), StandardCharsets.UTF_8))) {
            for (String l; (l = r.readLine()) != null; ) if (!l.trim().isEmpty()) out.add(l.trim());
        } catch (IOException e) { /* none */ }
        return out;
    }

    @Override
    public void onBackPressed() {
        if (mMain) { super.onBackPressed(); return; }
        showMain();
    }

    /* ------------------------------------------------------------------ screens */

    private TextView text(String s, int size, int color) { return Ui.text(mRoot, s, size, color); }

    private View button(String s, View.OnClickListener l) { return Ui.button(mRoot, s, l); }

    private void showMain() {
        mMain = true;
        Ui.clear(mRoot);
        Ui.title(mRoot, "for Android");
        button("Play", v -> { startActivity(new Intent(this, GameActivity.class)); finish(); });
        /* Resolution: the height the game renders at (the width follows the screen's shape); the display scales it
           to the screen. Choices up to the screen's own height. */
        final android.content.SharedPreferences prefs = getSharedPreferences("settings", MODE_PRIVATE);
        final int[] heights = resolutions();
        final Ui.MenuButton res = Ui.button(mRoot, "Resolution", true, null);
        res.setNote(resolutionLabel(prefs.getInt("render_height", 540)));
        res.setOnClickListener(v -> {
            int cur = prefs.getInt("render_height", 540), next = heights[0];
            for (int k = 0; k < heights.length; k++) if (heights[k] == cur) { next = heights[(k + 1) % heights.length]; break; }
            prefs.edit().putInt("render_height", next).apply();
            res.setNote(resolutionLabel(next));
        });
        if (!bundled())   /* add, replace or delete the player's own game files (SetupActivity) */
            Ui.button(mRoot, "Game files", true, v -> { startActivity(new Intent(this, SetupActivity.class).putExtra(SetupActivity.EXTRA_MANAGE, true)); finish(); });
        Ui.button(mRoot, "Check for updates", true, v -> { startActivity(new Intent(this, UpdateActivity.class)); finish(); });
        /* GitHub's new-issue page in the browser, with the phone and app version filled in */
        Ui.button(mRoot, "Report a problem", true, v -> {
            String body = "**What happened:**\n\n\n**Phone:** " + android.os.Build.MANUFACTURER + " " + android.os.Build.MODEL
                        + "\n**Android:** " + android.os.Build.VERSION.RELEASE + "\n**App version:** " + UpdateActivity.currentVersion() + "\n";
            Uri u = Uri.parse("https://github.com/" + UpdateActivity.REPO + "/issues/new").buildUpon()
                       .appendQueryParameter("body", body).build();
            try { startActivity(new Intent(Intent.ACTION_VIEW, u)); }
            catch (android.content.ActivityNotFoundException e) { Toast.makeText(this, "No browser found", Toast.LENGTH_SHORT).show(); }
        });
        if (sUpdate != null) showUpdateNote();
        else if (!sChecked && UpdateActivity.due(this)) {   /* quietly, at most once a day */
            sChecked = true;
            new Thread(() -> {
                try {
                    UpdateActivity.Release r = UpdateActivity.check(this);
                    if (r != null) runOnUiThread(() -> { sUpdate = r; if (mMain) showUpdateNote(); });
                } catch (java.io.IOException e) { /* offline: try another day */ }
            }, "update-check").start();
        }
    }

    private static UpdateActivity.Release sUpdate;   /* found by the quiet check */
    private static boolean sChecked;

    private void showUpdateNote() {
        TextView t = Ui.text(mRoot, "Version " + sUpdate.version + " is available: tap Check for updates", 14, Ui.GOLD_LIGHT);
        t.setTypeface(t.getTypeface(), android.graphics.Typeface.BOLD);
    }

    /* ------------------------------------------------------------------ resolution */

    private int[] screenSize() {   /* landscape: {long side, short side} in pixels */
        android.util.DisplayMetrics m = new android.util.DisplayMetrics();
        getWindowManager().getDefaultDisplay().getRealMetrics(m);
        return new int[] { Math.max(m.widthPixels, m.heightPixels), Math.min(m.widthPixels, m.heightPixels) };
    }

    private int[] resolutions() {
        int[] all = { 480, 540, 720, 900, 1080, 1440 };
        int screenH = screenSize()[1], n = 0;
        for (int h : all) if (h <= screenH) n++;
        if (n == 0) return new int[] { 540 };
        int[] out = new int[n];
        for (int k = 0; k < n; k++) out[k] = all[k];
        return out;
    }

    /* "720p  1560x720", the width as the game picks it (host/core.c video_size: the screen's shape, 4:3 to 2.4:1) */
    private String resolutionLabel(int h) {
        int[] s = screenSize();
        int w = (int) (((long) h * s[0] / s[1] + 1) & ~1L);
        w = Math.max(w, h * 4 / 3); w = Math.min(w, h * 12 / 5); w &= ~1;
        return h + "p  " + w + "x" + h;
    }

    private boolean bundled() {
        try (InputStream in = getAssets().open("game_manifest.txt")) { return true; } catch (IOException e) { return false; }
    }
}
